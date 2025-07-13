document.addEventListener('DOMContentLoaded', () => {
    // --- STATE MANAGEMENT ---
    let currentUser = null; // Will hold { email, fullName, companyName }
    let chatHistory = [];
    let currentScheduleDetails = null; // To hold details from the bot for confirmation
    let indexingIntervalId = null; // --- NEW: To track the polling interval

    // --- DOM ELEMENT REFERENCES ---
    const userSetupModal = document.getElementById('user-setup-modal');
    const userSetupForm = document.getElementById('user-setup-form');
    const fullNameInput = document.getElementById('full-name-input');
    const emailInput = document.getElementById('email-input');
    const companyNameInput = document.getElementById('company-name-input');
    const loginBtn = document.getElementById('login-btn');
    const closeModalBtn = document.getElementById('close-modal-btn');

    const chatBox = document.getElementById('chat-box');
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    
    const scheduleBar = document.getElementById('schedule-confirmation-bar');
    const scheduleDetailsText = document.getElementById('schedule-details-text');

    const uploadBtn = document.getElementById('upload-btn');
    const documentUploadInput = document.getElementById('document-upload');
    const resetDataBtn = document.getElementById('reset-data-btn');
    const switchUserBtn = document.getElementById('switch-user-btn');

    const appContainer = document.querySelector('.chat-app-container');

    const API_BASE_URL = 'http://localhost:8000/api';

    // --- CORE FUNCTIONS ---

    const displayMessage = (sender, message) => {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender);

        if (sender === 'bot' && message === 'loading') {
            messageElement.classList.add('loading');
            messageElement.innerHTML = `<div class="content"><div class="typing-indicator"><span></span><span></span><span></span></div></div>`;
        } else {
            const contentElement = document.createElement('div');
            contentElement.classList.add('content');
            
            if (sender === 'bot') {
                // A simple markdown-to-HTML converter for bot messages
                const renderedHtml = message
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;")
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
                    .replace(/\n/g, '<br>'); // Newlines
                
                contentElement.innerHTML = renderedHtml;
            } else {
                contentElement.textContent = message; // Keep user messages as plain text
            }
            messageElement.appendChild(contentElement);
        }
        
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    };

    const setChatInputEnabled = (enabled) => {
        messageInput.disabled = !enabled;
        sendBtn.disabled = !enabled;
        if (enabled) {
            messageInput.placeholder = "Start typing...";
        } else {
            messageInput.placeholder = "Please wait, processing documents...";
        }
    };

    const handleSendMessage = async () => {
        const messageText = messageInput.value.trim();
        if (!messageText || !currentUser) return;

        displayMessage('user', messageText);
        messageInput.value = '';
        displayMessage('bot', 'loading');
        
        chatHistory.push({ role: 'user', content: messageText });

        try {
            const response = await fetch(`${API_BASE_URL}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: currentUser.email,
                    message: messageText,
                    history: chatHistory,
                }),
            });

            if (chatBox.lastChild && chatBox.lastChild.classList.contains('loading')) {
                chatBox.removeChild(chatBox.lastChild);
            }

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'An unknown error occurred.');
            }

            const data = await response.json();
            displayMessage('bot', data.answer);
            
            chatHistory.push({ role: 'assistant', content: data.answer });

            if (data.schedule_details) {
                currentScheduleDetails = data.schedule_details;
                scheduleDetailsText.textContent = `At ${currentScheduleDetails.address} on ${new Date(currentScheduleDetails.time).toLocaleString()}`;
                scheduleBar.style.display = 'flex';
            }

        } catch (error) {
            if (chatBox.lastChild && chatBox.lastChild.classList.contains('loading')) {
                chatBox.removeChild(chatBox.lastChild);
            }
            const errorMessage = `Error: ${error.message}`;
            displayMessage('bot', errorMessage);
            chatHistory.push({ role: 'assistant', content: errorMessage });
        }
    };

    const loadUserHistory = async (email) => {
        try {
            const response = await fetch(`${API_BASE_URL}/conversations/${email}`);
            if (!response.ok) {
                console.log(`No conversation history found for ${email}.`);
                return [];
            }
            const data = await response.json();
            return data.history || [];
        } catch (error) {
            console.error('Error fetching history:', error);
            displayMessage('bot', 'Could not retrieve your past conversations.');
            return [];
        }
    };

    const handleUserSetup = async (e) => {
        e.preventDefault();
        const fullName = fullNameInput.value.trim();
        const email = emailInput.value.trim();
        const companyName = companyNameInput.value.trim();

        if (!fullName || !email) {
            alert('Full Name and Email are required to sign up.');
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/user`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ full_name: fullName, email, company_name: companyName }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Could not sign up.');
            }

            const newUser = await response.json();
            currentUser = { 
                fullName: newUser.full_name, 
                email: newUser.email, 
                companyName: newUser.company ? newUser.company.name : ''
            };
            
            chatHistory = [];
            chatBox.innerHTML = '';
            userSetupModal.style.display = 'none';
            displayMessage('bot', `Welcome, ${currentUser.fullName}! How can I help you?`);

        } catch (error) {
            alert(`Sign-up failed: ${error.message}`);
        }
    };

    const handleLogin = async () => {
        const email = emailInput.value.trim();
        if (!email) {
            alert('Please enter an email address to log in.');
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/user?email=${encodeURIComponent(email)}`);

            if (response.status === 404) {
                alert('No user found with that email. Please sign up with your full name.');
                return;
            }

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Could not log in.');
            }

            const existingUser = await response.json();
            currentUser = {
                fullName: existingUser.full_name,
                email: existingUser.email,
                companyName: existingUser.company ? existingUser.company.name : '',
            };

            // --- REVISED LOGIC ---
            // 1. Fetch history first
            displayMessage('bot', 'loading'); // Show a loading indicator
            chatHistory = await loadUserHistory(currentUser.email);
            
            // 2. Clear the chatbox *once*
            chatBox.innerHTML = ''; 
            
            // 3. Render the history or a welcome message
            if (chatHistory.length > 0) {
                chatHistory.forEach(msg => displayMessage(msg.role === 'assistant' ? 'bot' : 'user', msg.content));
            } else {
                 displayMessage('bot', `Welcome back, ${currentUser.fullName}! How can I help you today?`);
            }
            
            userSetupModal.style.display = 'none';

            // --- Trigger document loading for the user ---
            await fetch(`${API_BASE_URL}/documents/load`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: currentUser.email }),
            });
            setChatInputEnabled(false); // Disable chat on login while docs load
            // Start polling for status updates
            // --- NEW: Store the interval ID ---
            if (indexingIntervalId) clearInterval(indexingIntervalId);
            indexingIntervalId = setInterval(() => checkIndexingStatus(indexingIntervalId), 3000);

        } catch (error) {
            alert(`Login failed: ${error.message}`);
        }
    };

    const handleSwitchUser = () => {
        // --- NEW: Clear all user-specific state ---
        currentUser = null;
        chatHistory = [];
        currentScheduleDetails = null;
        chatBox.innerHTML = '';
        scheduleBar.style.display = 'none';

        // --- NEW: Stop any active polling ---
        if (indexingIntervalId) {
            clearInterval(indexingIntervalId);
            indexingIntervalId = null;
        }

        // Now, show the modal with clean fields
        userSetupModal.style.display = 'flex';
        emailInput.value = '';
        fullNameInput.value = '';
        companyNameInput.value = '';
    };

    const handleScheduleConfirm = async () => {
        if (!currentScheduleDetails || !currentUser) return;

        try {
            const response = await fetch(`${API_BASE_URL}/schedule`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    email: currentUser.email,
                    address: currentScheduleDetails.address,
                    time: currentScheduleDetails.time,
                }),
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Could not schedule event.');
            }
            const data = await response.json();
            displayMessage('bot', `Great! I've scheduled the viewing. You can see the event here: ${data.event_url}`);

        } catch (error) {
            displayMessage('bot', `Scheduling failed: ${error.message}`);
        } finally {
            scheduleBar.style.display = 'none';
            currentScheduleDetails = null;
        }
    };
    
    const checkIndexingStatus = async (intervalId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/documents/status`);
            if (!response.ok) {
                // Stop polling on server error
                clearInterval(intervalId);
                indexingIntervalId = null; // --- NEW: Clear the stored ID
                displayMessage('bot', 'Could not retrieve indexing status from the server.');
                return;
            }

            const data = await response.json();
            
            // If the status is success, error, or idle the job is done.
            if (data.status === 'success' || data.status === 'error' || data.status === 'idle') {
                clearInterval(intervalId);
                indexingIntervalId = null; // --- NEW: Clear the stored ID
                // Don't show the "No documents" message if it's the default idle state
                if (data.message !== "No active indexing jobs.") {
                    displayMessage('bot', data.message);
                }
                setChatInputEnabled(true); // Re-enable chat
            } else {
                // You could update a persistent status message here if desired
                console.log(`Indexing status: ${data.status}`);
            }
        } catch (error) {
            clearInterval(intervalId);
            indexingIntervalId = null; // --- NEW: Clear the stored ID
            displayMessage('bot', `Error checking status: ${error.message}`);
        }
    };
    
    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (!file.name.endsWith('.csv')) {
            alert('Please upload a valid CSV file.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        // --- NEW: Add user_id to the form data ---
        if (currentUser && currentUser.email) {
            formData.append('user_id', currentUser.email);
        } else {
            alert("Could not identify the current user. Please try logging in again.");
            return;
        }

        displayMessage('bot', `Uploading ${file.name}...`);
        setChatInputEnabled(false); // Disable chat input

        try {
            const response = await fetch(`${API_BASE_URL}/documents/upload`, {
                method: 'POST',
                body: formData, // No 'Content-Type' header needed; browser sets it with boundary
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'File upload failed.');
            }

            const data = await response.json();
            displayMessage('bot', data.message); // e.g., "File received. Indexing will proceed..."
            
            // Start polling for status updates
            // --- NEW: Store the interval ID ---
            if (indexingIntervalId) clearInterval(indexingIntervalId);
            indexingIntervalId = setInterval(() => checkIndexingStatus(indexingIntervalId), 3000);

        } catch (error) {
            displayMessage('bot', `Error: ${error.message}`);
            setChatInputEnabled(true); // Re-enable on error
        }
    };

    const handleResetData = async () => {
        const promptMessage = currentUser 
            ? `Are you sure you want to reset all data for ${currentUser.email}?`
            : 'Are you sure you want to reset ALL application data? This cannot be undone.';
        
        if (confirm(promptMessage)) {
            try {
                const url = currentUser ? `${API_BASE_URL}/reset?user_id=${encodeURIComponent(currentUser.email)}` : `${API_BASE_URL}/reset`;
                const response = await fetch(url, { method: 'POST' });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Reset failed.');
                }
                
                const data = await response.json();
                displayMessage('bot', data.message);
                
                // If a global reset was performed, force a reload to restart the app
                if (!currentUser) {
                    setTimeout(() => window.location.reload(), 2000);
                } else {
                    handleSwitchUser(); // Go back to login screen for the user
                }
            } catch (error) {
                displayMessage('bot', `Error: ${error.message}`);
            }
        }
    };

    const handleCloseModal = () => {
        // Only allow closing the modal if a user is already logged in
        if (currentUser) {
            userSetupModal.style.display = 'none';
        }
    };

    // --- EVENT LISTENERS ---
    userSetupForm.addEventListener('submit', handleUserSetup);
    loginBtn.addEventListener('click', handleLogin);
    closeModalBtn.addEventListener('click', handleCloseModal);
    
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        handleSendMessage();
    });

    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });

    appContainer.addEventListener('click', (e) => {
        const target = e.target.closest('button');
        if (!target) return;

        const id = target.id;

        // The login and close buttons are now handled by direct listeners above.
        // We leave this block empty but structured to avoid breaking the else-if chain.
        if (id === 'login-btn' || id === 'close-modal-btn') {
            // Handled by direct listeners: loginBtn and closeModalBtn
        } else if (id === 'confirm-schedule-btn') {
            handleScheduleConfirm();
        } else if (id === 'cancel-schedule-btn') {
            scheduleBar.style.display = 'none';
            currentScheduleDetails = null;
        } else if (id === 'upload-btn') {
            if (!currentUser) {
                alert('Please log in or sign up before uploading documents.');
                userSetupModal.style.display = 'flex';
                return;
            }
            documentUploadInput.click();
        } else if (id === 'reset-data-btn') {
            handleResetData();
        } else if (id === 'switch-user-btn') {
            handleSwitchUser();
        }
    });
    
    documentUploadInput.addEventListener('change', handleFileUpload);
}); 