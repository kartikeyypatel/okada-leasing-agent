import React, { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, X } from 'lucide-react';
import { AnimatePresence, motion } from "framer-motion";

interface AIInputFieldProps {
  onSendMessage: (message: string) => void;
  onFileUpload: (files: File[]) => void;
  isLoading?: boolean;
}

const PLACEHOLDER_QUESTIONS = [
  "What are the lease terms?",
  "Can I schedule a property tour?",
  "What documents do I need to apply?",
  "Are utilities included in the rent?",
  "Show me available properties in Manhattan."
];

const AIInputField: React.FC<AIInputFieldProps> = ({ onSendMessage, onFileUpload, isLoading }) => {
  const [message, setMessage] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [currentPlaceholder, setCurrentPlaceholder] = useState(0);
  const intervalRef = useRef<any>(null);

  useEffect(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = setInterval(() => {
      setCurrentPlaceholder((prev) => (prev + 1) % PLACEHOLDER_QUESTIONS.length);
    }, 3000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px';
    }
  }, [message]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    if (files.length > 0) {
      setUploadedFiles(prev => [...prev, ...files]);
      onFileUpload(files);
    }
  };

  const removeFile = (name: string) => {
    setUploadedFiles(prev => prev.filter(file => file.name !== name));
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i)) + sizes[i];
  };

  const handleSubmit = () => {
    if ((message.trim() || uploadedFiles.length > 0) && !isLoading) {
      if (message.trim()) {
        onSendMessage(message);
      }
      setMessage('');
      setUploadedFiles([]);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="w-full max-w-full mx-auto p-0">
      {/* Main Input Container */}
      <div className={`relative transition-all duration-500 ease-out ${
        isFocused || message ? 'transform scale-100' : ''
      }`}>
        {/* Glow Effect */}
        <div className={`absolute inset-0 rounded-xl transition-all duration-500 pointer-events-none ${
          isFocused
            ? 'bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-pink-500/20 blur-md scale-100'
            : 'bg-gradient-to-r from-slate-200/50 via-slate-100/50 to-slate-200/50 blur-sm'
        }`}></div>

        {/* Input Container */}
        <div className={`relative backdrop-blur-md bg-white/80 border rounded-xl transition-all duration-300 ${
          isFocused
            ? 'border-blue-400/50 shadow-lg shadow-blue-500/10'
            : 'border-white/60 shadow-md shadow-slate-300/10'
        } hover:shadow-lg hover:shadow-slate-400/20`}>
          {/* Uploaded Files */}
          {uploadedFiles.length > 0 && (
            <div className="px-2 py-1 border-b border-white/30">
              <div className="flex flex-wrap gap-1">
                {uploadedFiles.map((file) => (
                  <div key={file.name} className="group flex items-center gap-1 bg-gradient-to-r from-slate-50/80 to-white/80 backdrop-blur-sm rounded-lg px-2 py-1 shadow-sm hover:shadow-md transition-all duration-200">
                    <div className="w-1.5 h-1.5 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"></div>
                    <span className="text-slate-700 font-medium text-xs truncate max-w-24">{file.name}</span>
                    <span className="text-slate-500 text-[10px]">({formatFileSize(file.size)})</span>
                    <button
                      onClick={() => removeFile(file.name)}
                      className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-500 transition-all duration-200 hover:scale-110"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
          {/* Main Input Area */}
          <div className="flex items-end p-2 gap-2">
            {/* Left Actions */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="group relative p-2 rounded-lg bg-gradient-to-br from-slate-100/80 to-white/80 hover:from-blue-100/80 hover:to-purple-100/80 transition-all duration-300 hover:scale-105 hover:shadow"
                title="Upload files"
              >
                <Paperclip className="w-4 h-4 text-slate-600 group-hover:text-blue-600 transition-colors duration-300" />
                <div className="absolute inset-0 rounded-lg bg-gradient-to-br from-blue-400/0 to-purple-400/0 group-hover:from-blue-400/20 group-hover:to-purple-400/20 transition-all duration-300"></div>
              </button>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                multiple
                className="hidden"
                accept=".txt,.pdf,.doc,.docx,.jpg,.jpeg,.png,.gif,.csv,.json"
              />
            </div>
            {/* Text Input */}
            <div className="flex-1 relative">
              <textarea
                ref={textareaRef}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={handleKeyDown}
                onFocus={() => setIsFocused(true)}
                onBlur={() => setIsFocused(false)}
                placeholder={''}
                className="w-full resize-none border-none outline-none text-slate-800 placeholder-slate-400 text-base leading-relaxed min-h-[28px] max-h-20 bg-transparent font-medium selection:bg-blue-200/50 py-1"
                rows={1}
                style={{ background: 'transparent' }}
                disabled={isLoading}
              />
              {/* Animated Placeholder Overlay */}
              {!(isFocused || message) && (
                <div className="pointer-events-none absolute left-0 top-0 w-full h-full flex items-center pl-2">
                  <AnimatePresence mode="wait">
                    <motion.p
                      key={`placeholder-${currentPlaceholder}`}
                      initial={{ y: 5, opacity: 0 }}
                      animate={{ y: 0, opacity: 1 }}
                      exit={{ y: -10, opacity: 0 }}
                      transition={{ duration: 0.3, ease: "linear" }}
                      className="text-slate-400 text-base font-normal truncate w-full"
                    >
                      {PLACEHOLDER_QUESTIONS[currentPlaceholder]}
                    </motion.p>
                  </AnimatePresence>
                </div>
              )}
              {/* Cursor Animation */}
              {isFocused && !message && (
                <div className="absolute top-1 left-0 w-0.5 h-6 bg-gradient-to-b from-blue-500 to-purple-500 animate-pulse rounded-full"></div>
              )}
            </div>
            {/* Send Button */}
            <button
              onClick={handleSubmit}
              disabled={(!message.trim() && uploadedFiles.length === 0) || isLoading}
              className={`group relative p-2 rounded-lg font-medium transition-all duration-300 ${
                message.trim() || uploadedFiles.length > 0
                  ? 'bg-gradient-to-br from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white shadow-md shadow-blue-500/20 hover:shadow-lg hover:shadow-purple-500/30 hover:scale-105 transform-gpu'
                  : 'bg-gradient-to-br from-slate-200/80 to-slate-300/80 text-slate-400 cursor-not-allowed'
              }`}
              title="Send message"
            >
              <Send className="w-5 h-5" />
              {(message.trim() || uploadedFiles.length > 0) && (
                <div className="absolute inset-0 rounded-lg bg-gradient-to-br from-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export { AIInputField };
export default AIInputField; 