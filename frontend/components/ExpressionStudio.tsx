'use client'

import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Wand2, Film, Trash2, Loader2, RotateCcw, Sparkles } from 'lucide-react'
import { toast } from 'react-hot-toast'
import { api } from '@/lib/api'
import Image from 'next/image'
import type { Avatar, ExpressionPhoto, ExpressionParams } from '@/lib/types'

const DEFAULT_PARAMS: ExpressionParams = {
  rotate_pitch: 0, rotate_yaw: 0, rotate_roll: 0,
  blink: 0, eyebrow: 0, wink: 0, pupil_x: 0, pupil_y: 0,
  aaa: 0, eee: 0, woo: 0, smile: 0,
  src_ratio: 1, crop_factor: 1.7,
}

// Ranges transcribed from the actual ExpressionEditor node (see
// backend/app/schemas.py's ExpressionParams / model_experiments/comfyui/
// expression_editor_client.py's module docstring) — not guessed.
const SLIDER_GROUPS: { title: string; sliders: { key: keyof ExpressionParams; label: string; min: number; max: number; step: number }[] }[] = [
  {
    title: 'Head pose',
    sliders: [
      { key: 'rotate_pitch', label: 'Pitch (nod)', min: -20, max: 20, step: 0.5 },
      { key: 'rotate_yaw', label: 'Yaw (turn)', min: -20, max: 20, step: 0.5 },
      { key: 'rotate_roll', label: 'Roll (tilt)', min: -20, max: 20, step: 0.5 },
    ],
  },
  {
    title: 'Eyes',
    sliders: [
      { key: 'blink', label: 'Blink', min: -20, max: 5, step: 0.5 },
      { key: 'eyebrow', label: 'Eyebrow', min: -10, max: 15, step: 0.5 },
      { key: 'wink', label: 'Wink (right eye)', min: 0, max: 25, step: 0.5 },
      { key: 'pupil_x', label: 'Pupil X', min: -15, max: 15, step: 0.5 },
      { key: 'pupil_y', label: 'Pupil Y', min: -15, max: 15, step: 0.5 },
    ],
  },
  {
    title: 'Mouth',
    sliders: [
      { key: 'aaa', label: 'Open ("ah")', min: -30, max: 120, step: 1 },
      { key: 'eee', label: 'Pull ("ee")', min: -20, max: 15, step: 0.2 },
      { key: 'woo', label: 'Pucker ("oo")', min: -20, max: 15, step: 0.2 },
      { key: 'smile', label: 'Smile ↔ frown', min: -0.3, max: 1.3, step: 0.01 },
    ],
  },
  {
    title: 'Advanced',
    sliders: [
      { key: 'src_ratio', label: 'Keep original expression', min: 0, max: 1, step: 0.05 },
      { key: 'crop_factor', label: 'Face crop zoom', min: 1.5, max: 2.5, step: 0.05 },
    ],
  },
]

// Quick-start points, not fixed outputs — every slider stays adjustable
// after picking one. Same values used for the original 6 idle-segment
// expressions, kept here as starting points rather than the only options.
const QUICK_PRESETS: { label: string; params: Partial<ExpressionParams> }[] = [
  { label: 'Closed smile', params: { smile: 0.6, aaa: 0, eee: 0, woo: 0, rotate_pitch: 0 } },
  { label: 'Closed neutral', params: { smile: 0, aaa: 0, eee: 0, woo: 0, rotate_pitch: 0 } },
  { label: 'Closed frown', params: { smile: -0.3, aaa: 0, eee: 0, woo: 0, rotate_pitch: 0 } },
  { label: 'Open smile', params: { smile: 0.5, aaa: 30, eee: 15, woo: 0, rotate_pitch: 1.5 } },
  { label: 'Open neutral', params: { smile: 0, aaa: 50, eee: 0, woo: 0, rotate_pitch: 2.5 } },
  { label: 'Open frown', params: { smile: -0.25, aaa: 25, eee: 0, woo: 15, rotate_pitch: 1.25 } },
]

export function ExpressionStudio({ initialAvatarId }: { initialAvatarId?: string | null }) {
  const queryClient = useQueryClient()
  const { data: avatars, isLoading: avatarsLoading } = useQuery({
    queryKey: ['avatars'],
    queryFn: api.getAvatars,
  })
  const [avatarId, setAvatarId] = useState<string | null>(initialAvatarId ?? null)
  const [params, setParams] = useState<ExpressionParams>(DEFAULT_PARAMS)
  const [label, setLabel] = useState('')
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [selectedPhotoId, setSelectedPhotoId] = useState<string | null>(null)
  const [targetSlot, setTargetSlot] = useState(0)

  const avatar = avatars?.find((a: Avatar) => a.id === avatarId) ?? null

  const generateMutation = useMutation({
    mutationFn: () => api.generateExpression(avatarId as string, { params, label: label.trim() || undefined }),
    onSuccess: (updated: Avatar) => {
      toast.success('Expression generated', { icon: '🎭' })
      queryClient.invalidateQueries({ queryKey: ['avatars'] })
      const newest = updated.expression_photos?.[updated.expression_photos.length - 1]
      if (newest) setPreviewUrl(newest.url)
    },
    onError: () => toast.error('Failed to generate expression'),
  })

  const deleteMutation = useMutation({
    mutationFn: (photoId: string) => api.deleteExpressionPhoto(avatarId as string, photoId),
    onSuccess: () => {
      toast.success('Deleted')
      queryClient.invalidateQueries({ queryKey: ['avatars'] })
    },
    onError: () => toast.error('Failed to delete'),
  })

  const renderMutation = useMutation({
    mutationFn: ({ photoUrl, slotIndex }: { photoUrl: string; slotIndex: number }) =>
      api.renderIdleSegmentFromExpression(avatarId as string, photoUrl, slotIndex),
    onSuccess: () => {
      toast.success('Idle segment rendered', { icon: '🎬' })
      queryClient.invalidateQueries({ queryKey: ['avatars'] })
      setSelectedPhotoId(null)
    },
    onError: () => toast.error('Failed to render idle segment — this takes several minutes, check back if it timed out'),
  })

  const setParam = (key: keyof ExpressionParams, value: number) =>
    setParams((p) => ({ ...p, [key]: value }))

  if (avatarsLoading) {
    return <div className="text-center py-16 text-gray-500 text-sm">Loading avatars…</div>
  }

  if (!avatarId) {
    return (
      <div className="max-w-2xl mx-auto text-center py-16">
        <Sparkles size={32} className="mx-auto mb-3 text-primary-400" />
        <p className="text-white font-medium mb-1">Pick an avatar to edit expressions for</p>
        <p className="text-gray-500 text-sm mb-6">LivePortrait edits a still photo's expression — pick which avatar's photo to work from.</p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 max-w-lg mx-auto">
          {(avatars ?? []).map((a: Avatar) => (
            <button
              key={a.id}
              onClick={() => setAvatarId(a.id)}
              className="glass-card rounded-xl overflow-hidden hover:ring-1 hover:ring-primary-500/40 transition-all"
            >
              <div className="aspect-square relative bg-surface-700">
                {(a.thumbnail_url || a.image_url) && (
                  <Image src={(a.thumbnail_url || a.image_url) as string} alt={a.name} fill className="object-cover" />
                )}
              </div>
              <p className="text-xs text-white p-2 truncate">{a.name}</p>
            </button>
          ))}
        </div>
      </div>
    )
  }

  if (!avatar) return null

  return (
    <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-6">
      {/* ── Preview + avatar switcher ── */}
      <div className="space-y-3">
        <button onClick={() => setAvatarId(null)} className="text-xs text-gray-500 hover:text-white">
          ← Switch avatar
        </button>
        <div className="glass-card rounded-xl overflow-hidden aspect-square relative bg-surface-700">
          <Image
            src={previewUrl || avatar.thumbnail_url || avatar.image_url || ''}
            alt={avatar.name}
            fill
            className="object-cover"
          />
        </div>
        <p className="text-sm text-white font-medium">{avatar.name}</p>
        <p className="text-xs text-gray-500">
          {previewUrl ? 'Showing your latest generated expression.' : "Showing this avatar's current photo."}
        </p>

        <input
          type="text"
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          placeholder="Name this expression (optional)"
          className="w-full px-3 py-2 rounded-xl bg-surface-700/80 border border-white/10 text-white text-sm
                     placeholder:text-gray-600 focus:outline-none focus:ring-2 focus:ring-primary-500/50"
        />

        <button
          onClick={() => generateMutation.mutate()}
          disabled={generateMutation.isPending}
          className="btn-primary w-full text-sm py-2 rounded-xl flex items-center justify-center gap-2"
        >
          {generateMutation.isPending ? <Loader2 size={14} className="animate-spin" /> : <Wand2 size={14} />}
          Generate
        </button>
        <button
          onClick={() => setParams(DEFAULT_PARAMS)}
          className="btn-ghost w-full text-xs py-1.5 rounded-xl flex items-center justify-center gap-1.5"
        >
          <RotateCcw size={12} />
          Reset sliders
        </button>
      </div>

      {/* ── Controls ── */}
      <div className="space-y-5">
        <div>
          <p className="text-xs font-medium text-gray-400 mb-2">Quick start</p>
          <div className="flex flex-wrap gap-1.5">
            {QUICK_PRESETS.map((preset) => (
              <button
                key={preset.label}
                onClick={() => setParams((p) => ({ ...p, ...preset.params }))}
                className="btn-secondary text-[11px] px-2.5 py-1 rounded-lg"
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>

        {SLIDER_GROUPS.map((group) => (
          <div key={group.title} className="glass-card rounded-xl p-4">
            <p className="text-xs font-semibold text-primary-400 mb-3">{group.title}</p>
            <div className="space-y-3">
              {group.sliders.map((s) => (
                <div key={s.key}>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-xs text-gray-400">{s.label}</label>
                    <span className="text-xs text-gray-500 tabular-nums">{params[s.key].toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min={s.min}
                    max={s.max}
                    step={s.step}
                    value={params[s.key]}
                    onChange={(e) => setParam(s.key, Number(e.target.value))}
                    className="w-full accent-primary-500"
                  />
                </div>
              ))}
            </div>
          </div>
        ))}

        {/* ── Library ── */}
        <div>
          <p className="text-xs font-medium text-gray-400 mb-2">
            Saved expressions {avatar.expression_photos?.length ? `(${avatar.expression_photos.length})` : ''}
          </p>
          {!avatar.expression_photos?.length ? (
            <p className="text-xs text-gray-600">Nothing generated yet — dial in sliders above and hit Generate.</p>
          ) : (
            <div className="grid grid-cols-4 sm:grid-cols-6 gap-2">
              {avatar.expression_photos.map((photo: ExpressionPhoto) => (
                <div key={photo.id} className="relative group">
                  <button
                    onClick={() => {
                      setSelectedPhotoId(selectedPhotoId === photo.id ? null : photo.id)
                      setPreviewUrl(photo.url)
                    }}
                    className={`relative aspect-square w-full rounded-lg overflow-hidden border transition-all
                      ${selectedPhotoId === photo.id ? 'border-primary-500 ring-2 ring-primary-500/50' : 'border-white/10 hover:border-white/30'}`}
                    title={photo.label}
                  >
                    <Image src={photo.url} alt={photo.label} fill className="object-cover" />
                  </button>
                  <button
                    onClick={() => deleteMutation.mutate(photo.id)}
                    disabled={deleteMutation.isPending}
                    className="absolute top-1 right-1 w-5 h-5 rounded-full bg-red-600/80 backdrop-blur-sm
                               flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                    title="Delete"
                  >
                    <Trash2 size={10} className="text-white" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {selectedPhotoId && (() => {
            const photo = avatar.expression_photos?.find((p: ExpressionPhoto) => p.id === selectedPhotoId)
            if (!photo) return null
            return (
              <div className="flex items-center gap-2 mt-3 p-2.5 rounded-lg bg-surface-700/60 border border-white/10">
                <span className="text-[11px] text-gray-400 flex-1">
                  Render <span className="text-primary-400">{photo.label}</span> into idle slot
                </span>
                <select
                  value={targetSlot}
                  onChange={(e) => setTargetSlot(Number(e.target.value))}
                  className="text-[11px] bg-surface-800 border border-white/10 rounded px-1.5 py-1 text-white"
                >
                  {Array.from({ length: 6 }, (_, i) => (
                    <option key={i} value={i}>Segment {i + 1}</option>
                  ))}
                </select>
                <button
                  onClick={() => renderMutation.mutate({ photoUrl: photo.url, slotIndex: targetSlot })}
                  disabled={renderMutation.isPending}
                  className="btn-primary text-[11px] px-2.5 py-1 rounded-lg flex items-center gap-1"
                >
                  {renderMutation.isPending ? <Loader2 size={11} className="animate-spin" /> : <Film size={11} />}
                  Render
                </button>
              </div>
            )
          })()}
          {renderMutation.isPending && (
            <p className="text-[11px] text-gray-500 mt-1.5">Rendering idle segment — this takes several minutes…</p>
          )}
        </div>
      </div>
    </div>
  )
}
