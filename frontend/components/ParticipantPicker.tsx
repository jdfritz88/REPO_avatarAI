'use client'

import { useQuery } from '@tanstack/react-query'
import { Bot, Check, Users } from 'lucide-react'
import { api } from '@/lib/api'
import type { Participant } from '@/lib/types'

interface ParticipantPickerProps {
  selectedIds: string[]
  onChange: (ids: string[]) => void
}

const TYPE_LABEL: Record<Participant['type'], string> = {
  kindroid: 'Kindroid',
  openai_compat: 'Mistral',
  anthropic: 'Claude',
}

/**
 * Multi-select picker for optional AI participants (Kindroid kins, Mistral,
 * etc.) beyond the primary avatar — same toggle/ring/checkmark visual
 * language as AvatarList's single-select cards, but tracking an array.
 * Selecting none keeps the session on the original single-avatar pipeline.
 */
export function ParticipantPicker({ selectedIds, onChange }: ParticipantPickerProps) {
  const { data: participants, isLoading } = useQuery({
    queryKey: ['participants'],
    queryFn: api.listParticipants,
  })

  const toggle = (id: string) => {
    onChange(selectedIds.includes(id) ? selectedIds.filter((x) => x !== id) : [...selectedIds, id])
  }

  if (isLoading) {
    return (
      <div className="card flex flex-col gap-3">
        <div className="h-4 skeleton rounded w-1/3" />
        <div className="h-16 skeleton rounded" />
      </div>
    )
  }

  if (!participants || participants.length === 0) {
    return null // no multi-agent participants configured — nothing to show
  }

  return (
    <div className="card flex flex-col gap-4">
      <div>
        <div className="flex items-center gap-2">
          <Users size={16} className="text-primary-400" />
          <h2 className="text-lg font-bold text-white">Group Participants</h2>
        </div>
        <p className="text-sm text-gray-500 mt-0.5">
          Optional — add other AI participants to talk alongside your avatar. Leave empty for a
          normal one-on-one chat.
        </p>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        {participants.map((p: Participant) => {
          const isSelected = selectedIds.includes(p.id)
          return (
            <button
              key={p.id}
              type="button"
              onClick={() => toggle(p.id)}
              aria-pressed={isSelected}
              className={`relative flex flex-col items-start gap-1.5 px-3 py-2.5 rounded-xl text-left
                border transition-all duration-200
                ${isSelected
                  ? 'border-primary-500 bg-primary-500/10 shadow-glow-sm'
                  : 'border-white/10 bg-surface-700/60 hover:border-primary-500/40 hover:bg-surface-700'
                }`}
            >
              <div className="flex items-center gap-1.5 w-full">
                <Bot size={13} className={isSelected ? 'text-primary-400' : 'text-gray-500'} />
                <span className="font-semibold text-sm text-white truncate">{p.name}</span>
                {isSelected && (
                  <Check size={13} className="text-primary-400 ml-auto flex-shrink-0" />
                )}
              </div>
              <span className="text-[10px] text-gray-500">{TYPE_LABEL[p.type]}</span>
            </button>
          )
        })}
      </div>

      {selectedIds.length > 0 && (
        <p className="text-xs text-primary-300">
          {selectedIds.length} participant{selectedIds.length !== 1 ? 's' : ''} selected — round-robin
          mode by default (changeable in chat).
        </p>
      )}
    </div>
  )
}
