"""
Reliable Windows process-tree lifecycle management for GPU worker subprocesses.

Why this exists (confirmed with real evidence, not assumed):
On this machine, the backend venv's python.exe was created against the
Microsoft Store Python distribution, which is a launcher stub that re-execs
the real interpreter as a CHILD process — confirmed earlier via `wmic
process get ParentProcessId` showing the PID we spawn is the parent of a
second PID actually running the model. `taskkill /PID <pid> /T /F` (the
previous fix for this) walks the CURRENT process tree from that PID at the
moment it runs — which fails, silently, in a real and repeatable way: if
the stub has already exited on its own by the time a later timeout fires
the kill (a common, even typical launcher-shim pattern — the stub's only
job is to hand off and get out of the way), `taskkill` can't resolve the
now-dead stub PID to walk its descendants at all, so it never reaches the
grandchild — the process actually holding GPU memory — and that failure
was never being checked (the old code discarded taskkill's exit code
entirely). This was confirmed live: an orphaned worker pair survived
multiple subsequent timeout-triggered kill attempts in the same session.

The fix: Windows Job Objects. Any process assigned to a job, and any
further descendants IT spawns (via standard CreateProcess, which is what
Python's subprocess/asyncio machinery uses), automatically become part of
the same job — inheritance that doesn't depend on the immediate parent
still being alive at kill time. Setting JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
means closing the job handle terminates every process that was ever
assigned to it, in one guaranteed step, regardless of whether intermediate
processes in the chain have already exited. This is the standard,
documented Windows mechanism for this exact problem (used by Docker
Desktop, Chromium, and CPython's own `subprocess` module internals for
comparable cases) — implemented here via ctypes against kernel32.dll
directly so it needs no new dependency (pywin32 is not installed in this
venv).
"""
import ctypes
import logging
import os
from ctypes import wintypes
from typing import Optional

logger = logging.getLogger(__name__)

IS_WINDOWS = os.name == "nt"

if IS_WINDOWS:
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _JobObjectExtendedLimitInformation = 9
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _PROCESS_ALL_ACCESS = 0x1F0FFF

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD
    ]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.TerminateJobObject.restype = wintypes.BOOL
    _kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]


def create_kill_on_close_job(label: str) -> Optional[int]:
    """
    Creates a Windows Job Object configured so that closing its handle (or
    this process exiting) kills every process ever assigned to it. Returns
    the job handle, or None on non-Windows / on failure (callers must treat
    None as "job objects unavailable, fall back to taskkill only").
    """
    if not IS_WINDOWS:
        return None
    job = _kernel32.CreateJobObjectW(None, None)
    if not job:
        err = ctypes.get_last_error()
        logger.warning(f"[{label}] CreateJobObjectW failed (error {err}); "
                        f"will rely on taskkill /T /F only for this worker.")
        return None

    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = _kernel32.SetInformationJobObject(
        job, _JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
    )
    if not ok:
        err = ctypes.get_last_error()
        logger.warning(f"[{label}] SetInformationJobObject failed (error {err}); "
                        f"closing this job handle would NOT auto-kill its processes.")
        _kernel32.CloseHandle(job)
        return None
    return job


def assign_pid_to_job(job: int, pid: int, label: str) -> bool:
    """
    Assigns a running process (by PID) to an existing job object. Must be
    called as soon as possible after spawning that process — any further
    children IT spawns inherit job membership automatically, which is what
    makes this reliable even after the immediate process itself exits.
    Returns True on success; logs and returns False on failure rather than
    failing silently, so a failure here is visible instead of only showing
    up later as an unexplained orphan.
    """
    if not IS_WINDOWS or not job:
        return False
    hproc = _kernel32.OpenProcess(_PROCESS_ALL_ACCESS, False, pid)
    if not hproc:
        err = ctypes.get_last_error()
        logger.warning(f"[{label}] OpenProcess({pid}) failed (error {err}); "
                        f"cannot assign this worker to its kill-on-close job.")
        return False
    ok = _kernel32.AssignProcessToJobObject(job, hproc)
    _kernel32.CloseHandle(hproc)  # the job keeps its own reference; this handle isn't needed after assignment
    if not ok:
        err = ctypes.get_last_error()
        logger.warning(f"[{label}] AssignProcessToJobObject(pid={pid}) failed (error {err}).")
        return False
    return True


def kill_job(job: Optional[int], label: str) -> bool:
    """
    Terminates every process ever assigned to this job (including any
    grandchildren spawned after assignment, and even if the originally
    assigned process has already exited on its own) and closes the handle.
    Returns True if the job existed and TerminateJobObject reported success.
    """
    if not IS_WINDOWS or not job:
        return False
    ok = bool(_kernel32.TerminateJobObject(job, 1))
    if not ok:
        err = ctypes.get_last_error()
        logger.warning(f"[{label}] TerminateJobObject failed (error {err}).")
    _kernel32.CloseHandle(job)
    return ok
