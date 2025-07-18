import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';

interface UserProfileProps {
  user: { fullName: string, email: string, companyName?: string } | null;
  onClose: () => void;
  onLogin: (email: string) => void;
  onSignup: (email: string, fullName: string, companyName?: string) => void;
  onUpdate: (fullName: string, companyName: string) => void;
  onSwitchUser: () => void;
  onDeleteUser: () => void;
}

const UserProfile: React.FC<UserProfileProps> = ({ user, onClose, onLogin, onSignup, onUpdate, onSwitchUser, onDeleteUser }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [fullName, setFullName] = useState('');
  const [companyName, setCompanyName] = useState('');

  const handleUpdate = () => {
    if (user) {
      onUpdate(fullName || user.fullName, companyName || user.companyName || '');
    }
  };
  
  if (user) {
    return (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center">
            <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.9 }} className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-8 w-full max-w-md relative">
                <button onClick={onClose} className="absolute top-4 right-4 text-gray-500 hover:text-gray-800 dark:hover:text-white">
                    <X />
                </button>
                <h2 className="text-2xl font-bold mb-6 text-center text-gray-800 dark:text-white">Profile</h2>
                <div className="space-y-4">
                    <input type="text" placeholder="Full Name" defaultValue={user.fullName} onChange={(e) => setFullName(e.target.value)} className="w-full p-3 bg-gray-100 dark:bg-gray-700 rounded-lg"/>
                    <input type="email" placeholder="Email" value={user.email} readOnly className="w-full p-3 bg-gray-200 dark:bg-gray-600 rounded-lg cursor-not-allowed"/>
                    <input type="text" placeholder="Company Name" defaultValue={user.companyName} onChange={(e) => setCompanyName(e.target.value)} className="w-full p-3 bg-gray-100 dark:bg-gray-700 rounded-lg"/>
                </div>
                <div className="flex justify-between items-center mt-6">
                    <button onClick={handleUpdate} className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700">Update</button>
                    <button onClick={onSwitchUser} className="text-sm text-gray-600 dark:text-gray-400 hover:underline">Switch User</button>
                </div>
                 <button onClick={onDeleteUser} className="w-full mt-4 text-sm text-red-500 hover:underline">Delete Account</button>
            </motion.div>
        </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center">
      <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.9 }} className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-8 w-full max-w-md relative">
        <button onClick={onClose} className="absolute top-4 right-4 text-gray-500 hover:text-gray-800 dark:hover:text-white"><X /></button>
        <h2 className="text-2xl font-bold mb-6 text-center text-gray-800 dark:text-white">{isLogin ? 'Login' : 'Sign Up'}</h2>
        <div className="space-y-4">
          {!isLogin && <input type="text" placeholder="Full Name" value={fullName} onChange={(e) => setFullName(e.target.value)} className="w-full p-3 bg-gray-100 dark:bg-gray-700 rounded-lg"/>}
          <input type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} className="w-full p-3 bg-gray-100 dark:bg-gray-700 rounded-lg"/>
          {!isLogin && <input type="text" placeholder="Company Name (Optional)" value={companyName} onChange={(e) => setCompanyName(e.target.value)} className="w-full p-3 bg-gray-100 dark:bg-gray-700 rounded-lg"/>}
        </div>
        <button onClick={() => isLogin ? onLogin(email) : onSignup(email, fullName, companyName)} className="w-full mt-6 bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700">{isLogin ? 'Login' : 'Sign Up'}</button>
        <p className="text-center mt-4 text-sm">
          {isLogin ? "Don't have an account?" : "Already have an account?"}
          <button onClick={() => setIsLogin(!isLogin)} className="text-blue-500 hover:underline ml-1">{isLogin ? 'Sign Up' : 'Login'}</button>
        </p>
      </motion.div>
    </div>
  );
};

export default UserProfile;