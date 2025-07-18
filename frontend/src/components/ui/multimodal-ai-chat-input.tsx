// /frontend/src/components/ui/multimodal-ai-chat-input.tsx
'use client';

import React, { useRef, useState, useEffect } from 'react';
import type { KeyboardEvent, ChangeEvent } from 'react';
import { Loader2 as LoaderIcon, X as XIcon, Paperclip, Send } from 'lucide-react';

type PureMultimodalInputProps = {
  chatId: string;
  isGenerating: boolean;
  onSend: (payload: { input: string; attachments: File[] }) => void;
  onStop: () => void;
  attachments: File[];
  setAttachments: (files: File[]) => void;
};

export const PureMultimodalInput: React.FC<PureMultimodalInputProps> = ({
  chatId,
  isGenerating,
  onSend,
  onStop,
  attachments,
  setAttachments,
}) => {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [input]);

  const handleInputChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (input.trim() || attachments.length > 0) {
        handleSend();
      }
    }
  };

  const handleSend = () => {
    if (!input.trim() && attachments.length === 0) return;
    onSend({ input: input.trim(), attachments });
    setInput('');
    setAttachments([]);
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      setAttachments([...attachments, ...files]);
      e.target.value = '';
    }
  };

  const handleRemoveAttachment = (idx: number) => {
    setAttachments(attachments.filter((_, i) => i !== idx));
  };

  return (
    <div className="w-full flex flex-col gap-2">
      {/* Attachments preview */}
      {attachments.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-1">
          {attachments.map((file, idx) => (
            <div key={idx} className="flex items-center bg-gray-100 rounded px-2 py-1 text-xs text-gray-700">
              <span className="truncate max-w-[120px]">{file.name}</span>
              <button
                type="button"
                className="ml-1 text-gray-400 hover:text-red-500"
                onClick={() => handleRemoveAttachment(idx)}
                aria-label={`Remove ${file.name}`}
              >
                <XIcon size={14} />
              </button>
            </div>
          ))}
        </div>
      )}
      <div className="flex items-end gap-2 w-full">
        {/* File upload button */}
        <button
          type="button"
          className="p-2 rounded-md border border-gray-300 bg-white hover:bg-gray-100 text-gray-600"
          onClick={() => fileInputRef.current?.click()}
          aria-label="Attach file"
          disabled={isGenerating}
        >
          <Paperclip size={18} />
        </button>
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          multiple
          onChange={handleFileChange}
          aria-label="Upload file"
        />
        {/* Textarea */}
        <textarea
          ref={textareaRef}
          className="flex-1 resize-none rounded-md border border-gray-300 bg-white px-3 py-2 text-base placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100 disabled:text-gray-400"
          rows={1}
          value={input}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          disabled={isGenerating}
          aria-label="Chat message input"
        />
        {/* Send/Stop button */}
        {isGenerating ? (
          <button
            type="button"
            className="p-2 rounded-md bg-red-100 text-red-600 hover:bg-red-200 border border-red-200"
            onClick={onStop}
            aria-label="Stop generating"
          >
            <LoaderIcon className="animate-spin" size={18} />
          </button>
        ) : (
          <button
            type="button"
            className="p-2 rounded-md bg-blue-600 text-white hover:bg-blue-700 disabled:bg-gray-300 disabled:text-gray-400"
            onClick={handleSend}
            disabled={!input.trim() && attachments.length === 0}
            aria-label="Send message"
          >
            <Send size={18} />
          </button>
        )}
      </div>
    </div>
  );
}; 