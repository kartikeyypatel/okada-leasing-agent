import { useState } from 'react';
import toast from 'react-hot-toast';
import { X, LogOut, Trash2 } from 'lucide-react';
import { motion } from 'framer-motion';

// --- TYPE DEFINITIONS ---
interface User {
    email: string;
    fullName: string;
    companyName?: string;
}

interface UserProfileProps {
    user: User | null;
    onClose: () => void;
    onLogin: (email: string) => void;
    onSignup: (email: string, fullName: string, companyName?: string) => void;
    onUpdate: (fullName: string, companyName: string) => void;
    onSwitchUser: () => void;
    onDeleteUser: () => void;
}

// --- SUB-COMPONENTS for Login and Profile Views ---

const AuthView = ({ onLogin, onSignup, onClose }: Pick<UserProfileProps, 'onLogin' | 'onSignup' | 'onClose'>) => {
    const [email, setEmail] = useState('');
    const [fullName, setFullName] = useState('');
    const [companyName, setCompanyName] = useState('');

    return (
        <div className="p-6">
            <h2 className="text-2xl font-bold text-center mb-2">Welcome</h2>
            <p className="text-center text-sm text-gray-500 mb-6">Log in or create an account to begin.</p>
            <form onSubmit={(e) => { e.preventDefault(); onSignup(email, fullName, companyName); }}>
                <div className="space-y-4">
                    <input type="text" value={fullName} onChange={(e) => setFullName(e.target.value)} placeholder="Full Name (for sign-up)" className="w-full p-2 border border-gray-300 rounded-md"/>
                    <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="your.email@company.com" required className="w-full p-2 border border-gray-300 rounded-md"/>
                    <input type="text" value={companyName} onChange={(e) => setCompanyName(e.target.value)} placeholder="Company Name (optional)" className="w-full p-2 border border-gray-300 rounded-md"/>
                </div>
                <div className="flex items-center gap-3 mt-6">
                    <button type="button" onClick={() => onLogin(email)} className="w-full bg-gray-200 text-gray-800 py-2 rounded-md hover:bg-gray-300 transition-colors">Log In</button>
                    <button type="submit" className="w-full bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700 transition-colors">Sign Up</button>
                </div>
            </form>
        </div>
    );
};

const ProfileView = ({ user, onUpdate, onSwitchUser, onDeleteUser }: Pick<UserProfileProps, 'user' | 'onUpdate' | 'onSwitchUser' | 'onDeleteUser'>) => {
    if (!user) return null;
    const [fullName, setFullName] = useState(user.fullName);
    const [companyName, setCompanyName] = useState(user.companyName || '');

    return (
        <div className="p-6">
            <div className="flex items-center gap-4 mb-6">
                <div className="w-16 h-16 bg-blue-100 text-blue-600 flex items-center justify-center rounded-full text-2xl font-bold">
                    {user.fullName.charAt(0).toUpperCase()}
                </div>
                <div>
                    <h2 className="text-2xl font-bold">{user.fullName}</h2>
                    <p className="text-gray-500">{user.email}</p>
                </div>
            </div>

            <form onSubmit={(e) => { e.preventDefault(); onUpdate(fullName, companyName); }}>
                <h3 className="text-lg font-semibold mb-3">Update Your Details</h3>
                <div className="space-y-4">
                    <input type="text" value={fullName} onChange={(e) => setFullName(e.target.value)} className="w-full p-2 border border-gray-300 rounded-md"/>
                    <input type="text" value={companyName} onChange={(e) => setCompanyName(e.target.value)} placeholder="Company Name" className="w-full p-2 border border-gray-300 rounded-md"/>
                </div>
                <button type="submit" className="w-full mt-4 bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700">Save Changes</button>
            </form>

            <div className="mt-6 border-t pt-4 flex flex-col items-start space-y-2">
                <button onClick={onSwitchUser} className="flex items-center gap-2 text-gray-600 hover:text-black p-1">
                    <LogOut size={16} /> Switch User / Log Out
                </button>
                <button onClick={onDeleteUser} className="flex items-center gap-2 text-red-600 hover:text-red-800 p-1">
                    <Trash2 size={16} /> Delete Account
                </button>
            </div>
        </div>
    );
};

// --- MAIN MODAL COMPONENT ---
const UserProfile = ({ user, onClose, onLogin, onSignup, onUpdate, onSwitchUser, onDeleteUser }: UserProfileProps) => {
    return (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="bg-white rounded-lg shadow-xl w-full max-w-md relative"
            >
                <button onClick={onClose} className="absolute top-3 right-3 text-gray-400 hover:text-gray-700 z-10">
                    <X size={24}/>
                </button>
                
                {user ? (
                    <ProfileView 
                        user={user} 
                        onUpdate={onUpdate} 
                        onSwitchUser={onSwitchUser} 
                        onDeleteUser={onDeleteUser} 
                    />
                ) : (
                    <AuthView 
                        onLogin={onLogin} 
                        onSignup={onSignup} 
                        onClose={onClose} 
                    />
                )}
            </motion.div>
        </div>
    );
};

export default UserProfile;