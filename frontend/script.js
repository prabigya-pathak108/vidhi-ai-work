let currentSessionId = null;
let currentUserId = null;

async function startChat() {
    const userId = document.getElementById('userId').value.trim();
    if (!userId) {
        alert("Please enter a User ID");
        return;
    }

    try {
        const userResponse = await fetch(`/api/users/create?user_id=${userId}&name=User_${userId}`, { 
            method: 'POST' 
        });
        
        if (userResponse.ok) {
            await userResponse.json();
        } else if (userResponse.status === 400) {
        } else {
            const errorData = await userResponse.json();
        }

        const response = await fetch(`/api/chat/session/${userId}`, { method: 'POST' });
        
        if (!response.ok) {
            const err = await response.json();
            alert(err.detail || "Failed to create session.");
            return;
        }

        const data = await response.json();
        
        currentSessionId = data.session_id;
        currentUserId = userId;

        document.getElementById('setup-overlay').classList.add('hidden');
        document.getElementById('chat-interface').classList.remove('hidden');

        loadHistory();
    } catch (error) {
        alert("Server connection failed. Is the server running?");
    }
}

async function sendMessage() {
    const input = document.getElementById('userInput');
    const content = input.value.trim();
    
    if (!content) {
        return;
    }
    
    if (!currentSessionId) {
        alert('No active session. Please refresh the page.');
        return;
    }

    input.value = '';
    appendMessage('human', content);

    const loadingMsgObj = appendMessage('system', '🤔 Vidhi is thinking...');
    loadingMsgObj.classList.add('loading-state');

    try {
        const payload = {
            session_id: currentSessionId,
            content: content
        };

        const response = await fetch('/api/chat/message', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorData = await response.json();
            loadingMsgObj.remove();
            appendMessage('system', `Error: ${errorData.detail || 'Unknown error occurred'}`);
            return;
        }

        const data = await response.json();
        
        loadingMsgObj.remove();

        const aiResponse = data.ai_response || 'No response generated';
        
        let sources = [];
        if (data.sources && Array.isArray(data.sources) && data.sources.length > 0) {
            sources = data.sources;
        }

        appendMessage('system', aiResponse, sources);
        
    } catch (error) {
        loadingMsgObj.remove();
        appendMessage('system', "❌ Error: Could not reach server. Please check if the server is running.");
    }
}

function appendMessage(role, text, sources = [], confidence = '') {
    const chatBox = document.getElementById('chat-box');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    
    let html = `<div class="bubble">
                    <div class="text">${escapeHtml(text)}</div>`;
    
    if (confidence && role === 'system') {
        const confidenceColor = confidence === 'high' ? '#22c55e' : confidence === 'medium' ? '#f59e0b' : '#ef4444';
        html += `<div style="margin-top: 8px;">
                    <span style="font-size: 0.7rem; background: ${confidenceColor}20; color: ${confidenceColor}; padding: 2px 8px; border-radius: 4px; font-weight: 600;">
                        Confidence: ${confidence.toUpperCase()}
                    </span>
                 </div>`;
    }
    
    if (sources && sources.length > 0) {
        html += `<div class="sources-container">
                    <span class="source-label">📚 Legal Sources:</span>
                    <div class="sources-list">
                        ${sources.slice(0, 3).map(s => `
                            <span class="source-tag">
                                <i data-lucide="book-open" style="width:12px; height:12px"></i> 
                                ${escapeHtml(s.substring(0, 30))}${s.length > 30 ? '...' : ''}
                            </span>`).join('')}
                    </div>
                    <button class="view-sources-btn" onclick="openSourcesModal(${JSON.stringify(sources).replace(/"/g, '&quot;')})">
                        <i data-lucide="external-link" style="width:14px; height:14px"></i>
                        View All ${sources.length} Source${sources.length > 1 ? 's' : ''}
                    </button>
                 </div>`;
    }
    
    html += `</div>`;
    msgDiv.innerHTML = html;
    
    chatBox.appendChild(msgDiv);
    
    if (window.lucide) lucide.createIcons();
    
    setTimeout(() => {
        chatBox.scrollTop = chatBox.scrollHeight;
    }, 100);
    
    return msgDiv; 
}

async function loadHistory() {
    if (!currentSessionId) {
        return;
    }

    try {
        const response = await fetch(`/api/chat/history/${currentSessionId}`);
        
        if (!response.ok) {
            return;
        }
        
        const messages = await response.json();
        
        const chatBox = document.getElementById('chat-box');
        chatBox.innerHTML = '';
        
        appendMessage('system', 'नमस्कार! I am Vidhi AI, your Nepali Legal Assistant. How can I help you today?');
        
        messages.forEach(msg => {
            appendMessage(msg.role, msg.content);
        });
        
    } catch (error) {
    }
}

function handleKeyPress(e) {
    if (e.key === 'Enter') {
        e.preventDefault();
        sendMessage();
    }
}

function openSourcesModal(sources) {
    const modal = document.getElementById('sourcesModal');
    const modalBody = document.getElementById('sourcesModalBody');
    
    modalBody.innerHTML = '';
    
    if (!sources || sources.length === 0) {
        modalBody.innerHTML = `
            <div class="no-sources-message">
                <i data-lucide="book-x" style="width:48px; height:48px; margin-bottom:12px; opacity:0.5"></i>
                <p>No sources available for this response.</p>
            </div>`;
    } else {
        sources.forEach((source, index) => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            sourceItem.innerHTML = `
                <div class="source-icon">
                    <i data-lucide="file-text"></i>
                </div>
                <div class="source-text">
                    <strong style="color: #60a5fa; display: block; margin-bottom: 4px;">Source ${index + 1}:</strong>
                    ${escapeHtml(source)}
                </div>
            `;
            modalBody.appendChild(sourceItem);
        });
    }
    
    modal.classList.add('active');
    
    if (window.lucide) lucide.createIcons();
}

function closeSourcesModal() {
    const modal = document.getElementById('sourcesModal');
    modal.classList.remove('active');
}

function closeModalOnOutsideClick(event) {
    if (event.target.id === 'sourcesModal') {
        closeSourcesModal();
    }
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeSourcesModal();
    }
})

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function displayAnswer(data) {
    const chatMessages = document.getElementById('chatMessages');
    
    const answerDiv = document.createElement('div');
    answerDiv.className = 'message bot-message';
    
    let answerHTML = `<div class="message-content">${data.answer}</div>`;
    
    if (data.sources && data.sources.length > 0) {
        answerHTML += `
            <div class="sources-section">
                <button class="view-sources-btn" onclick="showSourcesModal(${JSON.stringify(data.sources).replace(/"/g, '&quot;')})">
                    <svg width="16" height="16" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z"/>
                    </svg>
                    View Sources (${data.sources.length})
                </button>
            </div>
        `;
    }
    
    answerDiv.innerHTML = answerHTML;
    chatMessages.appendChild(answerDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showSourcesModal(sources) {
    const modal = document.getElementById('sourcesModal');
    const modalSourcesList = document.getElementById('modalSourcesList');
    
    modalSourcesList.innerHTML = '';
    
    sources.forEach((source, index) => {
        const sourceDiv = document.createElement('div');
        sourceDiv.className = 'source-item';
        
        sourceDiv.innerHTML = `
            <div class="source-header">
                <span class="source-number">${index + 1}</span>
                <h3>${source.act || 'Legal Document'}</h3>
            </div>
            <div class="source-details">
                ${source.act_english ? `<p class="source-meta"><strong>English:</strong> ${source.act_english}</p>` : ''}
                ${source.chapter ? `<p class="source-meta"><strong>Chapter:</strong> ${source.chapter}</p>` : ''}
                ${source.section ? `<p class="source-meta"><strong>Section:</strong> ${source.section}</p>` : ''}
                ${source.section_title ? `<p class="source-meta"><strong>Title:</strong> ${source.section_title}</p>` : ''}
                ${source.published_date ? `<p class="source-meta"><strong>Published:</strong> ${source.published_date}</p>` : ''}
            </div>
            <div class="source-text">
                <strong>Content:</strong>
                <pre>${source.text || 'No content available'}</pre>
            </div>
        `;
        
        modalSourcesList.appendChild(sourceDiv);
    });
    
    modal.style.display = 'block';
}

function closeSourcesModal() {
    const modal = document.getElementById('sourcesModal');
    modal.style.display = 'none';
}

window.onclick = function(event) {
    const modal = document.getElementById('sourcesModal');
    if (event.target == modal) {
        closeSourcesModal();
    }
}

document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        closeSourcesModal();
    }
});