// Chat Application JavaScript - Updated with SSE improvements
console.log('Chat.js loaded - Version with intermediate message support');
let sessionId = null;
let agentAvailable = false;
let currentEventSource = null;
let responseReceived = false;

function initializeChat(sId, available) {
    sessionId = sId;
    agentAvailable = available;
    
    if (agentAvailable) {
        setupEventHandlers();
        focusInput();
    }
}

function setupEventHandlers() {
    const form = document.getElementById('chat-form');
    const input = document.getElementById('message-input');
    
    form.addEventListener('submit', handleFormSubmit);
    input.addEventListener('keypress', handleKeyPress);
}

function handleFormSubmit(e) {
    e.preventDefault();
    
    const input = document.getElementById('message-input');
    const message = input.value.trim();
    
    if (message && agentAvailable) {
        sendMessage(message);
        input.value = '';
    }
}

function handleKeyPress(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        document.getElementById('chat-form').dispatchEvent(new Event('submit'));
    }
}

function sendMessage(message) {
    // Reset response tracking for new message
    responseReceived = false;
    
    // Add user message to chat
    addMessage(message, 'user');
    
    // Send message to backend
    const formData = new FormData();
    formData.append('message', message);
    formData.append('session_id', sessionId);
    
    fetch('/send_message', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success' && data.stream_url) {
            // Start listening to the SSE stream
            startSSEStream(data.stream_url);
        } else {
            throw new Error('Failed to start conversation');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        hideLoadingIndicator();
        addMessage('Sorry, I encountered an error. Please try again.', 'agent', 'error');
    });
}

function startSSEStream(streamUrl) {
    // Close existing EventSource if any
    if (currentEventSource) {
        currentEventSource.close();
    }
    
    // Create new EventSource for the proxy stream
    const proxyUrl = `/proxy_stream?stream_url=${encodeURIComponent(streamUrl)}`;
    currentEventSource = new EventSource(proxyUrl);
    
    let agentResponse = '';
    let isFirstMessage = true;
    
    currentEventSource.onmessage = function(event) {
        console.log('Raw SSE message:', event.data);
        try {
            const data = JSON.parse(event.data);
            console.log('Parsed SSE data:', data);
            handleSSEEvent(data, isFirstMessage);
            isFirstMessage = false;
        } catch (e) {
            console.error('Error parsing SSE data:', e);
            // Handle raw data
            handleRawSSEData(event.data);
        }
    };
    
    currentEventSource.onerror = function(event) {
        console.error('SSE Error:', event);
        if (currentEventSource) {
            currentEventSource.close();
            currentEventSource = null;
        }
        hideLoadingIndicator();
        
        // Only show error if we haven't received any response yet
        if (!responseReceived) {
            addMessage('❌ Sorry, I encountered an error processing your request.', 'agent', 'error');
        }
    };
}

function handleSSEEvent(data, isFirstMessage) {
    console.log('SSE Event received:', data);
    
    // Handle the nested structure from the proxy
    let eventType = data.event;
    let eventData = data.data;
    
    if (eventType) {
        switch (eventType) {
            case 'connected':
                console.log('Connected to agent stream');
                break;
            case 'processing_start':
                updateLoadingText('Processing your request...');
                break;
            case 'agent_thinking':
                updateLoadingText('Thinking about how to help you...');
                break;
            case 'registry_query_start':
                const query = eventData?.query || 'help with this request';
                console.log('Registry query start:', query);
                addMessage(`🔍 Searching for an agent that can ${query}...`, 'agent', 'progress');
                break;
            case 'registry_response':
                const agentCount = eventData?.agent_count || 0;
                console.log('Registry response - agent count:', agentCount);
                if (agentCount > 0) {
                    if (agentCount === 1) {
                        addMessage(`📋 Great! Found an agent that can help us.`, 'agent', 'progress');
                    } else {
                        addMessage(`📋 Excellent! Found ${agentCount} agents that can help us.`, 'agent', 'progress');
                    }
                } else {
                    addMessage(`📋 No agents found for this request.`, 'agent', 'progress');
                }
                break;
            case 'agent_call_start':
                const agentName = eventData?.agent_name || 'another agent';
                console.log('Agent call start:', agentName);
                addMessage(`📞 Passing your request to the ${agentName}...`, 'agent', 'progress');
                break;
            case 'agent_call_response':
                const responseAgent = eventData?.agent_name || 'agent';
                console.log('Agent call response from:', responseAgent);
                addMessage(`✅ The ${responseAgent} responded that it successfully completed your request!`, 'agent', 'progress');
                break;
            case 'agent_event':
                // Handle agent_event which can contain intermediate or final responses
                if (eventData?.content && eventData.content.trim()) {
                    console.log('Agent event content:', eventData.content, 'is_final:', eventData.is_final);
                    
                    if (eventData.is_final && !responseReceived) {
                        // Final response - show as normal chat message
                        console.log('Displaying final agent_event response:', eventData.content);
                        addMessage(eventData.content, 'agent');
                        responseReceived = true;
                    } else if (!eventData.is_final) {
                        // Intermediate response - show as thinking message
                        console.log('Displaying intermediate agent_event:', eventData.content);
                        addMessage(eventData.content, 'agent', 'thinking');
                    }
                }
                break;
            case 'final_response':
                // Handle final_response as backup only if no response received yet
                const response = eventData?.response || eventData?.message || '';
                if (response && !responseReceived) {
                    console.log('Displaying final_response as backup:', response);
                    addMessage(response, 'agent');
                    responseReceived = true;
                }
                break;
            case 'stream_end':
                if (currentEventSource) {
                    currentEventSource.close();
                    currentEventSource = null;
                }
                break;
            case 'error':
            case 'registry_error':
            case 'agent_call_error':
                const errorMsg = eventData?.message || 'An error occurred';
                addMessage(`❌ ${errorMsg}`, 'agent', 'error');
                responseReceived = true;
                break;
            case 'keepalive':
                // Handle keepalive - just log it
                console.log('Keepalive received');
                break;
            default:
                console.log('Unhandled event type:', eventType, eventData);
                break;
        }
    }
}

function handleRawSSEData(data) {
    // Handle raw SSE data that might not be JSON
    console.log('Raw SSE data:', data);
}

function addMessage(content, sender, type = 'normal') {
    const messagesContainer = document.getElementById('chat-messages');
    const messageWrapper = document.createElement('div');
    messageWrapper.className = `message-wrapper ${sender}-message`;
    
    const messageBubble = document.createElement('div');
    messageBubble.className = `message-bubble ${sender}-bubble`;
    
    if (type === 'error') {
        messageBubble.classList.add('error-bubble');
    } else if (type === 'thinking') {
        messageBubble.classList.add('thinking-bubble');
    } else if (type === 'progress') {
        messageBubble.classList.add('progress-bubble');
    }
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    if (sender === 'agent') {
        const icon = document.createElement('i');
        icon.className = 'fas fa-robot me-2';
        messageContent.appendChild(icon);
    } else {
        const icon = document.createElement('i');
        icon.className = 'fas fa-user me-2';
        messageContent.appendChild(icon);
    }
    
    const textNode = document.createTextNode(content);
    messageContent.appendChild(textNode);
    
    const messageTime = document.createElement('div');
    messageTime.className = 'message-time';
    messageTime.textContent = getCurrentTime();
    
    messageBubble.appendChild(messageContent);
    messageBubble.appendChild(messageTime);
    messageWrapper.appendChild(messageBubble);
    messagesContainer.appendChild(messageWrapper);
    
    // Scroll to bottom
    scrollToBottom();
}

function showLoadingIndicator() {
    console.log('showLoadingIndicator called');
    const loadingIndicator = document.getElementById('loading-indicator');
    if (loadingIndicator) {
        loadingIndicator.classList.remove('d-none');
        console.log('Loading indicator shown, classes:', loadingIndicator.className);
    } else {
        console.log('Error: loading-indicator element not found');
    }
}

function hideLoadingIndicator() {
    const loadingIndicator = document.getElementById('loading-indicator');
    loadingIndicator.classList.add('d-none');
}

function updateLoadingText(text) {
    console.log('updateLoadingText called with:', text);
    const loadingIndicator = document.getElementById('loading-indicator');
    const textSpan = loadingIndicator.querySelector('span');
    if (textSpan) {
        textSpan.textContent = text;
        console.log('Loading text updated to:', text);
    } else {
        console.log('Error: loading text span not found');
    }
}

function scrollToBottom() {
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function focusInput() {
    const input = document.getElementById('message-input');
    if (input) {
        input.focus();
    }
}

function getCurrentTime() {
    const now = new Date();
    return now.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit',
        hour12: false 
    });
}

// Make getCurrentTime available globally
window.getCurrentTime = getCurrentTime;

// Handle page visibility changes
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        // Page is hidden, potentially close SSE connection
        if (currentEventSource) {
            currentEventSource.close();
            currentEventSource = null;
        }
    } else {
        // Page is visible again, user might want to continue chatting
        focusInput();
    }
});

// Clean up on page unload
window.addEventListener('beforeunload', function() {
    if (currentEventSource) {
        currentEventSource.close();
        currentEventSource = null;
    }
});

// Auto-scroll behavior
function setupAutoScroll() {
    const chatMessages = document.getElementById('chat-messages');
    let isUserScrolling = false;
    let scrollTimeout;
    
    chatMessages.addEventListener('scroll', function() {
        isUserScrolling = true;
        clearTimeout(scrollTimeout);
        
        // Reset after 1 second of no scrolling
        scrollTimeout = setTimeout(() => {
            isUserScrolling = false;
        }, 1000);
    });
    
    // Override scrollToBottom to respect user scrolling
    const originalScrollToBottom = window.scrollToBottom;
    window.scrollToBottom = function() {
        if (!isUserScrolling) {
            originalScrollToBottom();
        }
    };
}

// Initialize auto-scroll when DOM is loaded
document.addEventListener('DOMContentLoaded', setupAutoScroll);