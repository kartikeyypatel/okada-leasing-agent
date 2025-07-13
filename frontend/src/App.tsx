"use client"

import React, { useState, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { MessageCircle, X, Send, Paperclip, RotateCcw, Users } from 'lucide-react'

// Utility function to merge class names
const cn = (...inputs: any[]) => {
  return inputs.filter(Boolean).join(' ')
}

// Type definition for a message
interface Message {
  id: string
  content: string
  sender: 'user' | 'bot'
}

// Spinner component for loading states
const Spinner = ({ size = 20, color = "#8f8f8f" }: { size?: number; color?: string }) => (
  <div style={{ width: size, height: size, position: 'relative' }}>
    {[...Array(12)].map((_, i) => (
      <div
        key={i}
        style={{
          position: 'absolute',
          height: '8%',
          width: '24%',
          left: '38%',
          top: '46%',
          backgroundColor: color,
          borderRadius: '5px',
          animation: 'spin 1.2s linear infinite',
          animationDelay: `${-1.2 + i * 0.1}s`,
          transformOrigin: '50% 150%',
          transform: `rotate(${i * 30}deg) translate(0, -146%)`,
        }}
      />
    ))}
    {/* The <style jsx> tag is not standard in Vite/React. 
        A regular <style> tag is used with a template literal for the keyframes. 
        This is a workaround to keep styles scoped if not using a CSS-in-JS library. */}
    <style>
      {`
        @keyframes spin {
          0%, 39% { opacity: 1; }
          40% { opacity: 0.15; }
          100% { opacity: 0.15; }
        }
      `}
    </style>
  </div>
);


// Button component
const Button = React.forwardRef<HTMLButtonElement, React.ButtonHTMLAttributes<HTMLButtonElement>>(
  ({ className, ...props }, ref) => {
  return (
    <button
      className={cn("inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50", className)}
      ref={ref}
      {...props}
    />
  )
})
Button.displayName = "Button"

// Input component
const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => {
    return (
      <input
        className={cn("flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50", className)}
        ref={ref}
        {...props}
      />
    )
  }
)
Input.displayName = "Input"

// Main Chatbot Component
const Chatbot: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false)
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Hello! Welcome to Okada & Company. How can I help you with your leasing needs today?',
      sender: 'bot',
    }
  ])
  const [inputValue, setInputValue] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  React.useEffect(() => {
    if (isOpen) {
        setTimeout(scrollToBottom, 100);
    }
  }, [messages, isOpen])

  const sendMessage = useCallback(async (content: string, sender: 'user' | 'bot' = 'user') => {
    if (!content.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: content,
      sender: sender,
    };
    
    setMessages(prev => [...prev, userMessage]);
    
    if(sender === 'user') {
        setInputValue('');
        setIsTyping(true);

        try {
          // NOTE: Replace with your actual API endpoint and user handling
          const response = await fetch('/api/chat', { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              user_id: "anonymous-user", // Replace with actual user ID
              message: content,
              history: messages.map(m => ({role: m.sender, content: m.content})),
            }),
          });

          if (!response.ok) {
            throw new Error('Network response was not ok');
          }
          
          const data = await response.json();
          
          const botMessage: Message = {
            id: (Date.now() + 1).toString(),
            content: data.answer || 'I am unable to respond at the moment.',
            sender: 'bot',
          };
          setMessages(prev => [...prev, botMessage]);

        } catch (error) {
          const errorMessage: Message = {
            id: (Date.now() + 1).toString(),
            content: "I'm having trouble connecting right now. Please try again later.",
            sender: 'bot',
          };
          setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsTyping(false);
        }
    }
  }, [messages]);

  const handleSendMessage = () => {
    sendMessage(inputValue, 'user');
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  }

  const resetChat = () => {
    setMessages([
      {
        id: '1',
        content: 'Chat reset. How can I assist you?',
        sender: 'bot',
      }
    ]);
  }

  const switchUser = () => {
    // In a real app, this would trigger a login/logout flow
    alert("User session has been reset.");
    resetChat();
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      sendMessage(`📎 You attached: ${file.name}`, 'user');
      // In a real app, you would now upload this file.
    }
  }

  return (
    <>
      <div className="fixed bottom-6 right-6 z-50">
        <motion.button
          onClick={() => setIsOpen(!isOpen)}
          className="w-16 h-16 bg-blue-600 hover:bg-blue-700 text-white rounded-full shadow-2xl flex items-center justify-center transition-transform duration-300 ease-in-out"
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
        >
          <AnimatePresence mode="wait">
            {isOpen ? (
              <motion.div key="close" initial={{ rotate: -90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: 90, opacity: 0 }} transition={{ duration: 0.3 }}>
                <X className="w-8 h-8" />
              </motion.div>
            ) : (
              <motion.div key="open" initial={{ rotate: 90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: -90, opacity: 0 }} transition={{ duration: 0.3 }}>
                <MessageCircle className="w-8 h-8" />
              </motion.div>
            )}
          </AnimatePresence>
        </motion.button>
      </div>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 50, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 50, scale: 0.9 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="fixed bottom-24 right-6 w-[calc(100%-48px)] max-w-sm h-[70vh] max-h-[600px] bg-white rounded-2xl shadow-2xl border border-gray-200/80 flex flex-col overflow-hidden"
          >
            {/* Header */}
            <div className="bg-gray-50 border-b border-gray-200 p-4 flex items-center justify-between flex-shrink-0">
                <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-blue-600 rounded-full flex items-center justify-center text-white font-bold text-lg">O</div>
                    <div>
                        <h3 className="font-semibold text-sm text-gray-800">Okada & Company</h3>
                        <p className="text-xs text-green-500">● Online</p>
                    </div>
                </div>
              <div className="flex items-center space-x-1">
                <Button onClick={switchUser} title="Switch User" className="h-8 w-8 text-gray-500 hover:bg-gray-200/80 rounded-full" > <Users className="w-4 h-4" /> </Button>
                <Button onClick={resetChat} title="Reset Chat" className="h-8 w-8 text-gray-500 hover:bg-gray-200/80 rounded-full" > <RotateCcw className="w-4 h-4" /> </Button>
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.map((message) => (
                <div key={message.id} className={cn("flex", message.sender === 'user' ? "justify-end" : "justify-start")}>
                  <div className={cn("max-w-[85%] px-4 py-2.5 rounded-2xl text-sm shadow-sm", message.sender === 'user' ? "bg-blue-600 text-white rounded-br-lg" : "bg-gray-100 text-gray-800 rounded-bl-lg")}>
                    {message.content}
                  </div>
                </div>
              ))}
              
              {isTyping && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 px-4 py-2.5 rounded-2xl rounded-bl-lg shadow-sm">
                    <Spinner size={16} color="#6b7280" />
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="p-3 border-t border-gray-200 bg-gray-50">
              <div className="relative flex items-center">
                <Input
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type your message..."
                  className="w-full rounded-full pl-10 pr-20 h-11 bg-white border-gray-300 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                />
                <div className="absolute left-3">
                  <input type="file" ref={fileInputRef} onChange={handleFileUpload} accept=".csv,.pdf" className="hidden" />
                  <Button onClick={() => fileInputRef.current?.click()} title="Attach File" className="h-8 w-8 text-gray-400 hover:text-gray-600 rounded-full hover:bg-gray-200/80" >
                    <Paperclip className="w-5 h-5" />
                  </Button>
                </div>
                <div className="absolute right-2">
                    <Button onClick={handleSendMessage} disabled={!inputValue.trim() || isTyping} className="h-9 w-16 rounded-full bg-blue-600 hover:bg-blue-700 text-white disabled:bg-blue-300" >
                        <Send className="w-4 h-4" />
                    </Button>
                </div>
              </div>
      </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

// Main Website Component
const OkadaCompanyWebsite: React.FC = () => {
  return (
    <div className="min-h-screen bg-white font-sans antialiased">
      {/* Background Image */}
      <div 
          className="absolute inset-0 bg-cover bg-center bg-no-repeat z-0"
          style={{
            backgroundImage: `linear-gradient(to bottom, rgba(255, 255, 255, 0.4), rgba(255, 255, 255, 1)), url('https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?q=80&w=2070')`,
            filter: 'blur(2px)'
          }}
        />

      <div className="relative z-10">
        {/* Header */}
        <header className="py-6">
          <div className="max-w-7xl mx-auto px-6 lg:px-8">
            <div className="flex justify-between items-center">
              <h1 className="text-xl font-bold text-gray-900 tracking-wider">OKADA & COMPANY</h1>
              <nav className="hidden md:flex space-x-10 text-sm font-medium">
                <a href="#" className="text-gray-600 hover:text-gray-900 transition-colors">Services</a>
                <a href="#" className="text-gray-600 hover:text-gray-900 transition-colors">Properties</a>
                <a href="#" className="text-gray-600 hover:text-gray-900 transition-colors">About Us</a>
              </nav>
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <main className="h-[calc(100vh-88px)] flex items-center">
          <div className="max-w-7xl mx-auto px-6 lg:px-8">
            <div className="max-w-2xl text-left">
              <motion.h1 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, ease: "easeOut" }}
                className="text-5xl md:text-6xl font-bold text-gray-900 leading-tight"
              >
                Find Your Perfect<br/>
                <span className="text-blue-600">Commercial Space</span>
              </motion.h1>
              
              <motion.p 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
                className="mt-6 text-lg text-gray-600 max-w-xl"
              >
                Okada & Company is your trusted partner in navigating the NYC real estate market. Let our AI assistant help you find the ideal property today.
              </motion.p>
            </div>
          </div>
        </main>
      </div>

      {/* Chatbot Component */}
      <Chatbot />
    </div>
  )
}

export default OkadaCompanyWebsite
