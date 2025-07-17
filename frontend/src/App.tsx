import { useState } from 'react';
import toast, { Toaster } from 'react-hot-toast';
import Header from "./components/Header";
import Hero from "./components/Hero";
import Chatbot from "./components/Chatbot";
import UserProfile from './components/UserProfile';
import { User as UserIcon } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

export interface User {
  email: string;
  fullName: string;
  companyName?: string;
}

function App() {
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [isProfileModalOpen, setIsProfileModalOpen] = useState(false);

  // --- USER AUTHENTICATION & MANAGEMENT FUNCTIONS ---

  const loginUser = async (email: string) => {
    if (!email) {
        toast.error("Please enter an email to log in.");
        return;
    }
    try {
        const response = await fetch(`${API_BASE_URL}/user?email=${encodeURIComponent(email)}`);
        if (response.status === 404) {
            toast.error("User not found. Please sign up first.");
            return;
        }
        if (!response.ok) throw new Error("Could not log in.");
        
        const user = await response.json();
        const userData = { email: user.email, fullName: user.full_name, companyName: user.company?.name };
        setCurrentUser(userData);
        setIsProfileModalOpen(false);
        toast.success(`Welcome back, ${user.full_name}!`);
    } catch (error) {
        toast.error(`Login failed: ${(error as Error).message}`);
    }
  };
  
  const signupUser = async (email: string, fullName: string, companyName?: string) => {
    if (!email || !fullName) {
        toast.error("Full Name and Email are required to sign up.");
        return;
    }
    try {
        const response = await fetch(`${API_BASE_URL}/user`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ full_name: fullName, email: email, company_name: companyName }),
        });
        if (!response.ok) throw new Error("Sign up failed.");
        
        const user = await response.json();
        const userData = { email: user.email, fullName: user.full_name, companyName: user.company?.name };
        setCurrentUser(userData);
        setIsProfileModalOpen(false);
        toast.success(`Welcome, ${user.full_name}!`);
    } catch (error) {
        toast.error(`Sign up failed: ${(error as Error).message}`);
    }
  };
  
  const switchUser = async () => {
    if (!currentUser) return;
    try {
        await fetch(`${API_BASE_URL}/logout`, { method: 'POST' });
        setCurrentUser(null);
        setIsProfileModalOpen(false);
        toast.success("You have been logged out.");
    } catch (error) {
        toast.error("Logout failed. Please try again.");
    }
  };

  const deleteUser = async () => {
    if (!currentUser) return;
    if (window.confirm("Are you sure you want to delete your account? This will permanently erase your data.")) {
        try {
            const response = await fetch(`${API_BASE_URL}/user?email=${encodeURIComponent(currentUser.email)}`, { method: 'DELETE' });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Failed to delete account.");
            }
            toast.success("Your account has been successfully deleted.");
            await switchUser();
        } catch (error) {
            toast.error((error as Error).message);
        }
    }
  };
  
  const updateUser = async (fullName: string, companyName: string) => {
      if (!currentUser) return;
      try {
            const response = await fetch(`${API_BASE_URL}/user`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email: currentUser.email, full_name: fullName, company_name: companyName }),
            });
            if (!response.ok) {
                 const errorData = await response.json();
                 throw new Error(errorData.detail || "Failed to update details.");
            }
            const updatedUserData = await response.json();
            
            setCurrentUser({ 
                email: updatedUserData.email, 
                fullName: updatedUserData.full_name, 
                companyName: updatedUserData.company?.name 
            });

            toast.success("Details updated successfully!");
            setIsProfileModalOpen(false);
        } catch (error) {
            toast.error((error as Error).message);
        }
  };

  return (
    <div className="relative min-h-screen w-full">
      <div className="absolute inset-0 bg-cover bg-center filter blur-sm" style={{ backgroundImage: "url('/nyc-background.jpg')" }}></div>
      <div className="absolute inset-0 bg-black/40"></div>
      
      <div className="relative z-10 flex flex-col min-h-screen">
        <Header>
            {currentUser ? (
                <button 
                    onClick={() => setIsProfileModalOpen(true)}
                    className="flex items-center gap-2 text-white bg-white/10 hover:bg-white/20 px-4 py-2 rounded-lg transition-colors"
                >
                    <UserIcon size={18} />
                    <span>{currentUser.fullName}</span>
                </button>
            ) : (
                 <button 
                    onClick={() => setIsProfileModalOpen(true)}
                    className="text-white bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg transition-colors"
                >
                    Login / Sign Up
                </button>
            )}
        </Header>
        <main className="flex-grow flex items-center justify-center">
          <Hero />
        </main>
      </div>
      
      <Chatbot
        currentUser={currentUser}
        openAuthModal={() => setIsProfileModalOpen(true)}
      />
      
      {isProfileModalOpen && (
        <UserProfile 
            user={currentUser}
            onClose={() => setIsProfileModalOpen(false)}
            onLogin={loginUser}
            onSignup={signupUser}
            onUpdate={updateUser}
            onSwitchUser={switchUser}
            onDeleteUser={deleteUser}
        />
      )}
      <Toaster position="top-center" />
    </div>
  );
}

export default App;
