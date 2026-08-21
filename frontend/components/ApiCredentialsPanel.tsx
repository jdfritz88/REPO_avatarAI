'use client'

import { useMemo, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Plug, Plus, Trash2, Star, Loader2, CheckCircle2, XCircle, GripVertical,
  ArrowDownAZ, Pencil, X, Check,
} from 'lucide-react'
import { toast } from 'react-hot-toast'
import { api } from '@/lib/api'
import type { LlmCredential } from '@/lib/types'

const PROVIDER_LABELS: Record<string, string> = {
  anthropic: 'Anthropic (Claude)',
  mistral: 'Mistral',
  openai: 'OpenAI',
  kindroid: 'Kindroid',
  custom: 'Custom',
}

const KNOWN_PROVIDERS = ['anthropic', 'mistral', 'openai', 'kindroid', 'custom']

type SortMode = 'manual' | 'alpha' | 'favorite'

function StatusBadge({ cred }: { cred: LlmCredential }) {
  if (cred.last_verify_ok === null || cred.last_verify_ok === undefined) {
    return <span className="text-[11px] text-gray-500">Not tested</span>
  }
  if (cred.last_verify_ok) {
    return (
      <span className="text-[11px] text-green-400 flex items-center gap-1">
        <CheckCircle2 size={12} /> Connected
      </span>
    )
  }
  return (
    <span className="text-[11px] text-red-400 flex items-center gap-1" title={cred.last_verify_message || ''}>
      <XCircle size={12} /> Failed
    </span>
  )
}

export function ApiCredentialsPanel() {
  const queryClient = useQueryClient()
  const [sortMode, setSortMode] = useState<SortMode>('manual')
  const dragId = useRef<string | null>(null)
  const [manualOrder, setManualOrder] = useState<string[] | null>(null)

  const { data: credentials, isLoading } = useQuery({
    queryKey: ['llm-credentials'],
    queryFn: api.listLlmCredentials,
  })
  const { data: defaultBaseUrls } = useQuery({
    queryKey: ['llm-default-base-urls'],
    queryFn: api.getLlmDefaultBaseUrls,
    staleTime: Infinity,
  })

  const connectMutation = useMutation({
    mutationFn: (id: string) => api.connectLlmCredential(id),
    onSuccess: (updated) => {
      queryClient.invalidateQueries({ queryKey: ['llm-credentials'] })
      if (updated.last_verify_ok) toast.success(updated.last_verify_message || 'Connected', { icon: '🔌' })
      else toast.error(updated.last_verify_message || 'Connection failed')
    },
    onError: () => toast.error('Failed to test connection'),
  })

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.deleteLlmCredential(id),
    onSuccess: () => {
      toast.success('Removed')
      queryClient.invalidateQueries({ queryKey: ['llm-credentials'] })
    },
    onError: () => toast.error('Failed to remove'),
  })

  const favoriteMutation = useMutation({
    mutationFn: ({ id, is_favorite }: { id: string; is_favorite: boolean }) =>
      api.updateLlmCredential(id, { is_favorite }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['llm-credentials'] }),
    onError: () => toast.error('Failed to update'),
  })

  const relabelMutation = useMutation({
    mutationFn: ({ id, label }: { id: string; label: string }) => api.updateLlmCredential(id, { label }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['llm-credentials'] }),
    onError: () => toast.error('Failed to rename'),
  })

  const rekeyMutation = useMutation({
    mutationFn: ({ id, api_key }: { id: string; api_key: string }) => api.updateLlmCredential(id, { api_key }),
    onSuccess: () => {
      toast.success('Key updated — click Connect to verify')
      queryClient.invalidateQueries({ queryKey: ['llm-credentials'] })
    },
    onError: () => toast.error('Failed to update key'),
  })

  const reorderMutation = useMutation({
    mutationFn: (orderedIds: string[]) => api.reorderLlmCredentials(orderedIds),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['llm-credentials'] }),
    onError: () => toast.error('Failed to save new order'),
  })

  const createMutation = useMutation({
    mutationFn: api.createLlmCredential,
    onSuccess: (created) => {
      queryClient.invalidateQueries({ queryKey: ['llm-credentials'] })
      if (created.last_verify_ok) toast.success('Added and connected', { icon: '✅' })
      else toast(created.last_verify_message || 'Added, but connection test failed', { icon: '⚠️' })
      setNewLabel(''); setNewKey(''); setNewBaseUrl(''); setNewKinId(''); setAdding(false)
    },
    onError: () => toast.error('Failed to add credential'),
  })

  // ── sort ──────────────────────────────────────────────────────────────
  const sorted = useMemo(() => {
    if (!credentials) return []
    if (sortMode === 'alpha') return [...credentials].sort((a, b) => a.label.localeCompare(b.label))
    if (sortMode === 'favorite') {
      return [...credentials].sort((a, b) => Number(b.is_favorite) - Number(a.is_favorite) || a.label.localeCompare(b.label))
    }
    // manual: server sort_order, unless a drag just reordered locally
    const base = [...credentials].sort((a, b) => a.sort_order - b.sort_order)
    if (!manualOrder) return base
    const byId = new Map(base.map((c) => [c.id, c]))
    return manualOrder.map((id) => byId.get(id)).filter((c): c is LlmCredential => !!c)
  }, [credentials, sortMode, manualOrder])

  const onDragStart = (id: string) => { dragId.current = id }
  const onDragOver = (e: React.DragEvent, overId: string) => {
    e.preventDefault()
    if (sortMode !== 'manual' || !dragId.current || dragId.current === overId) return
    const current = manualOrder ?? sorted.map((c) => c.id)
    const from = current.indexOf(dragId.current)
    const to = current.indexOf(overId)
    if (from === -1 || to === -1) return
    const next = [...current]
    next.splice(from, 1)
    next.splice(to, 0, dragId.current)
    setManualOrder(next)
  }
  const onDragEnd = () => {
    if (manualOrder) reorderMutation.mutate(manualOrder)
    dragId.current = null
  }

  // ── add-new row ───────────────────────────────────────────────────────
  const [adding, setAdding] = useState(false)
  const [newProvider, setNewProvider] = useState('anthropic')
  const [newLabel, setNewLabel] = useState('')
  const [newKey, setNewKey] = useState('')
  const [newBaseUrl, setNewBaseUrl] = useState('')
  const [newKinId, setNewKinId] = useState('')

  const submitNew = () => {
    if (!newLabel.trim() || !newKey.trim()) {
      toast.error('Label and API key are required')
      return
    }
    if (newProvider === 'custom' && !newBaseUrl.trim()) {
      toast.error('Custom providers need an API link (base URL)')
      return
    }
    if (newProvider === 'kindroid' && !newKinId.trim()) {
      toast.error('Kindroid needs a kin AI ID')
      return
    }
    createMutation.mutate({
      provider: newProvider,
      label: newLabel.trim(),
      api_key: newKey.trim(),
      api_base_url: newBaseUrl.trim() || undefined,
      kindroid_ai_id: newProvider === 'kindroid' ? newKinId.trim() : undefined,
    })
  }

  return (
    <div className="card flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white">API Keys</h2>
          <p className="text-sm text-gray-500 mt-0.5">
            Connect the AI providers your avatars use to talk — each key is tested for real when you click Connect.
          </p>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => { setSortMode('alpha'); setManualOrder(null) }}
            className={`btn-icon ${sortMode === 'alpha' ? 'text-primary-400 border-primary-500/30' : ''}`}
            title="Sort A–Z"
          >
            <ArrowDownAZ size={14} />
          </button>
          <button
            onClick={() => { setSortMode('favorite'); setManualOrder(null) }}
            className={`btn-icon ${sortMode === 'favorite' ? 'text-primary-400 border-primary-500/30' : ''}`}
            title="Sort by favorite"
          >
            <Star size={14} />
          </button>
        </div>
      </div>

      <div className="divider" />

      {isLoading ? (
        <div className="h-24 skeleton rounded" />
      ) : sorted.length === 0 ? (
        <p className="text-sm text-gray-500 text-center py-6">No API keys yet — add one below.</p>
      ) : (
        <div className="flex flex-col gap-2">
          {sorted.map((cred) => (
            <CredentialRow
              key={cred.id}
              cred={cred}
              draggable={sortMode === 'manual'}
              onDragStart={() => onDragStart(cred.id)}
              onDragOver={(e) => onDragOver(e, cred.id)}
              onDragEnd={onDragEnd}
              onConnect={() => connectMutation.mutate(cred.id)}
              connecting={connectMutation.isPending && connectMutation.variables === cred.id}
              onDelete={() => { if (window.confirm(`Remove "${cred.label}"?`)) deleteMutation.mutate(cred.id) }}
              onToggleFavorite={() => favoriteMutation.mutate({ id: cred.id, is_favorite: !cred.is_favorite })}
              onRelabel={(label) => relabelMutation.mutate({ id: cred.id, label })}
              onRekey={(key) => rekeyMutation.mutate({ id: cred.id, api_key: key })}
            />
          ))}
        </div>
      )}

      <div className="divider" />

      {!adding ? (
        <button onClick={() => setAdding(true)} className="btn-secondary self-start flex items-center gap-1.5 text-sm">
          <Plus size={14} /> Add API key
        </button>
      ) : (
        <div className="glass-card rounded-xl p-4 flex flex-col gap-3 border border-primary-500/30">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-gray-400">Provider</label>
              <select
                value={newProvider}
                onChange={(e) => { setNewProvider(e.target.value); setNewBaseUrl('') }}
                className="input-field"
              >
                {KNOWN_PROVIDERS.map((p) => (
                  <option key={p} value={p}>{PROVIDER_LABELS[p]}</option>
                ))}
              </select>
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-gray-400">Label</label>
              <input
                type="text"
                value={newLabel}
                onChange={(e) => setNewLabel(e.target.value)}
                placeholder={newProvider === 'kindroid' ? "e.g. Megumi's kin" : PROVIDER_LABELS[newProvider]}
                className="input-field"
              />
            </div>
            <div className="space-y-1.5 sm:col-span-2">
              <label className="text-xs font-medium text-gray-400">API key</label>
              <input
                type="password"
                value={newKey}
                onChange={(e) => setNewKey(e.target.value)}
                placeholder="Paste the API key"
                className="input-field"
                autoComplete="off"
              />
            </div>
            <div className="space-y-1.5 sm:col-span-2">
              <label className="text-xs font-medium text-gray-400">
                API link {newProvider === 'custom' ? '(required)' : '(optional — defaults to the standard endpoint)'}
              </label>
              <input
                type="text"
                value={newBaseUrl}
                onChange={(e) => setNewBaseUrl(e.target.value)}
                placeholder={defaultBaseUrls?.[newProvider] || 'https://api.example.com/v1'}
                className="input-field"
              />
            </div>
            {newProvider === 'kindroid' && (
              <div className="space-y-1.5 sm:col-span-2">
                <label className="text-xs font-medium text-gray-400">Kindroid kin AI ID</label>
                <input
                  type="text"
                  value={newKinId}
                  onChange={(e) => setNewKinId(e.target.value)}
                  placeholder="From Kindroid's API & advanced integrations settings"
                  className="input-field"
                />
              </div>
            )}
          </div>
          <div className="flex items-center justify-end gap-2">
            <button onClick={() => setAdding(false)} className="btn-ghost text-sm px-3 py-1.5">Cancel</button>
            <button
              onClick={submitNew}
              disabled={createMutation.isPending}
              className="btn-primary text-sm px-4 py-1.5 rounded-lg flex items-center gap-1.5"
            >
              {createMutation.isPending ? <Loader2 size={13} className="animate-spin" /> : <Plug size={13} />}
              Add &amp; connect
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function CredentialRow({
  cred, draggable, onDragStart, onDragOver, onDragEnd, onConnect, connecting,
  onDelete, onToggleFavorite, onRelabel, onRekey,
}: {
  cred: LlmCredential
  draggable: boolean
  onDragStart: () => void
  onDragOver: (e: React.DragEvent) => void
  onDragEnd: () => void
  onConnect: () => void
  connecting: boolean
  onDelete: () => void
  onToggleFavorite: () => void
  onRelabel: (label: string) => void
  onRekey: (key: string) => void
}) {
  const [editingLabel, setEditingLabel] = useState(false)
  const [labelDraft, setLabelDraft] = useState(cred.label)
  const [editingKey, setEditingKey] = useState(false)
  const [keyDraft, setKeyDraft] = useState('')

  return (
    <div
      draggable={draggable}
      onDragStart={onDragStart}
      onDragOver={onDragOver}
      onDragEnd={onDragEnd}
      className="flex items-center gap-2 px-3 py-2.5 rounded-xl bg-surface-700/60 border border-white/10"
    >
      {draggable && <GripVertical size={14} className="text-gray-600 cursor-grab flex-shrink-0" />}

      <button onClick={onToggleFavorite} className="flex-shrink-0" title="Favorite">
        <Star size={14} className={cred.is_favorite ? 'text-amber-400 fill-amber-400' : 'text-gray-600'} />
      </button>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          {editingLabel ? (
            <input
              autoFocus
              value={labelDraft}
              onChange={(e) => setLabelDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') { onRelabel(labelDraft); setEditingLabel(false) }
                if (e.key === 'Escape') { setLabelDraft(cred.label); setEditingLabel(false) }
              }}
              className="bg-surface-800 border border-white/10 rounded px-1.5 py-0.5 text-sm text-white w-40"
            />
          ) : (
            <span className="text-sm font-semibold text-white truncate">{cred.label}</span>
          )}
          {editingLabel ? (
            <>
              <button onClick={() => { onRelabel(labelDraft); setEditingLabel(false) }} className="text-green-400"><Check size={12} /></button>
              <button onClick={() => { setLabelDraft(cred.label); setEditingLabel(false) }} className="text-gray-500"><X size={12} /></button>
            </>
          ) : (
            <button onClick={() => setEditingLabel(true)} className="text-gray-600 hover:text-gray-300"><Pencil size={11} /></button>
          )}
          <span className="text-[10px] text-gray-500">{PROVIDER_LABELS[cred.provider] || cred.provider}</span>
        </div>
        <div className="flex items-center gap-2 mt-0.5">
          {editingKey ? (
            <>
              <input
                type="password"
                autoFocus
                value={keyDraft}
                onChange={(e) => setKeyDraft(e.target.value)}
                placeholder="New API key"
                className="bg-surface-800 border border-white/10 rounded px-1.5 py-0.5 text-xs text-white w-40"
              />
              <button onClick={() => { if (keyDraft.trim()) onRekey(keyDraft.trim()); setKeyDraft(''); setEditingKey(false) }} className="text-green-400"><Check size={12} /></button>
              <button onClick={() => { setKeyDraft(''); setEditingKey(false) }} className="text-gray-500"><X size={12} /></button>
            </>
          ) : (
            <>
              <span className="text-xs text-gray-500 font-mono">{cred.api_key_masked}</span>
              <button onClick={() => setEditingKey(true)} className="text-gray-600 hover:text-gray-300"><Pencil size={10} /></button>
            </>
          )}
          <StatusBadge cred={cred} />
        </div>
      </div>

      <button
        onClick={onConnect}
        disabled={connecting}
        className="btn-secondary text-xs px-2.5 py-1.5 rounded-lg flex items-center gap-1 flex-shrink-0"
      >
        {connecting ? <Loader2 size={12} className="animate-spin" /> : <Plug size={12} />}
        Connect
      </button>
      <button onClick={onDelete} className="text-gray-600 hover:text-red-400 flex-shrink-0" title="Remove">
        <Trash2 size={14} />
      </button>
    </div>
  )
}
