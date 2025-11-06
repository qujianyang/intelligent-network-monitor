// Initialize chatbot functionality

document.addEventListener('DOMContentLoaded', function() {

    const chatbotContainer = document.getElementById('chatbot-container');
    const chatbotToggle = document.getElementById('chatbot-toggle');
    const chatbotWindow = document.getElementById('chatbot-window');
    const chatbotInput = document.getElementById('chatbot-input');
    const chatbotSend = document.getElementById('chatbot-send');
    const chatbotMessages = document.getElementById('chatbot-messages');
    const typingIndicator = document.getElementById('typing-indicator');
    const charCounter = document.getElementById('char-counter');
    // Removed global status variables - we get fresh references in functions instead
    // const aiStatusIndicator = document.getElementById('ai-status-indicator');
    // const aiStatusText = document.getElementById('ai-status-text');
    const resizeHandle = document.getElementById('chatbot-resize-handle');
    const hourglassContainer = document.querySelector('.hourglass-container');
    const thoughtProcessContainer = document.getElementById('thought-process-container');
    if (hourglassContainer) hourglassContainer.style.display = 'none';

    let isOpen = false;
    let aiStatus = null;
    let statusCheckInProgress = false;

    // Staged image storage
    let stagedImage = null;  // Stores File object for upload

    // Thought process variables
    let thoughtProcessInterval = null;
    let currentThoughtStep = 0;

    // Resize functionality variables
    let isResizing = false;
    let startX, startY, startWidth, startHeight;

    // Tools dropdown variables
    const toolsBtn = document.getElementById('tools-btn');
    const toolsMenu = document.getElementById('tools-menu');

    // Mode management variables - Make globally accessible
    window.currentMode = 'chat'; // Default to normal chat mode
    window.modeConfig = {
        chat: {
            placeholder: "Chat with the AI assistant...",
            welcome: " Hello! I'm your AI assistant. Ask me anything!",
            icon: "fas fa-comments"
        },
        sop: {
            placeholder: "Ask about SOPs, procedures, or documentation...",
            welcome: " SOP Mode: I'll search procedures and documentation for you.",
            icon: "fas fa-book"
        },
        agent: {
            placeholder: "Ask complex questions requiring reasoning and tools...",
            welcome: " Agent Mode: I can use tools and reasoning to solve complex problems.",
            icon: "fas fa-robot"
        }
    };

    // Define the thought process steps based on backend logic
    const thoughtSteps = {
        'sop': [
            { text: "Searching knowledge base...", duration: 2000 },
            { text: "Analyzing relevant documents...", duration: 2500 },
            { text: "Synthesizing answer...", duration: 3000 },
            { text: "Formatting response...", duration: 1500 }
        ],
        'chat': [
            { text: "Understanding your question...", duration: 1500 },
            { text: "Generating response...", duration: 3000 },
            { text: "Finalizing answer...", duration: 1500 }
        ],
        'agent': [
            { text: "Analyzing request...", duration: 2000 },
            { text: "Selecting appropriate tools...", duration: 2500 },
            { text: "Executing reasoning process...", duration: 3500 },
            { text: "Compiling results...", duration: 2000 }
        ]
    };

    // ======================================
    // MODE MANAGEMENT FUNCTIONALITY
    // ======================================

    // Mode switching function
    window.setMode = function(mode) {
        // Validate mode
        if (!window.modeConfig || !window.modeConfig[mode]) {
            console.error('Invalid mode:', mode);
            return;
        }

        if (mode === window.currentMode) {
            return;
        }

        const previousMode = window.currentMode;
        window.currentMode = mode;

        try {
            // Update button states
            document.querySelectorAll('.mode-btn').forEach(btn => {
                btn.classList.remove('active');
            });

            const activeBtn = document.getElementById(mode + '-mode-btn');
            if (activeBtn) {
                activeBtn.classList.add('active');
            }

            // Update input placeholder
            const input = document.getElementById('chatbot-input');
            if (input) {
                input.placeholder = window.modeConfig[mode].placeholder;
            }

            // Show/hide document filter based on mode
            const filterContainer = document.getElementById('document-filter-container');
            if (filterContainer) {
                if (mode === 'sop') {
                    filterContainer.style.display = 'flex';
                    // Update dropdown when switching to SOP mode
                    updateDocumentFilter();
                } else {
                    filterContainer.style.display = 'none';
                }
            }

            // Show visual feedback
            showModeChangeToast(mode);

        } catch (error) {
            console.error('Error in setMode:', error);
            window.currentMode = previousMode;
        }
    };

    // Show mode change toast notification
    function showModeChangeToast(mode) {
        const modeNames = { chat: 'Chat Mode', sop: 'SOP Mode', agent: 'Agent Mode' };
        const toast = document.createElement('div');
        toast.style.cssText = `
            position: fixed; top: 80px; right: 20px; z-index: 10000;
            background: #28a745; color: white; padding: 10px 20px;
            border-radius: 5px; font-size: 14px; box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        `;
        toast.textContent = `Switched to ${modeNames[mode]}`;
        document.body.appendChild(toast);
        setTimeout(() => document.body.removeChild(toast), 2000);
    }

    window.addModeChangeMessage = function(mode) {
        try {
            const modeNames = { chat: 'Chat Mode', sop: 'SOP Mode', agent: 'Agent Mode' };
            const message = `Switched to ${modeNames[mode]}. ${window.modeConfig[mode].welcome}`;

            // Check if addMessage function is available
            if (typeof addMessage === 'function') {
                addMessage(message, 'system', {
                    confidence: 1.0,
                    architecture: mode + '_mode',
                    timestamp: new Date()
                });
            }
        } catch (error) {
            console.error('Error adding mode change message:', error);
        }
    };

    // ======================================
    // THOUGHT PROCESS FUNCTIONALITY
    // ======================================

    // Function to start thought process animation
    function startThoughtProcess(mode = 'chat') {
        const steps = thoughtSteps[mode] || thoughtSteps['chat'];
        currentThoughtStep = 0;

        // Show the thought process container
        if (thoughtProcessContainer) {
            thoughtProcessContainer.style.display = 'block';
        }

        // Hide the old hourglass
        if (hourglassContainer) {
            hourglassContainer.style.display = 'none';
        }

        // Start the step progression
        progressThroughSteps(steps);
    }

    // Function to progress through thought steps
    function progressThroughSteps(steps) {
        if (currentThoughtStep >= steps.length) {
            // Loop back to first step if we run out
            currentThoughtStep = 0;
        }

        const currentStep = steps[currentThoughtStep];
        const thoughtText = document.getElementById('thought-process-text');

        // Update text
        if (thoughtText) {
            thoughtText.textContent = currentStep.text;
        }

        // Update step indicators
        updateStepIndicators(currentThoughtStep, steps.length);

        // Schedule next step
        thoughtProcessInterval = setTimeout(() => {
            currentThoughtStep++;
            progressThroughSteps(steps);
        }, currentStep.duration);
    }

    // Function to update step indicator dots
    function updateStepIndicators(activeStep, totalSteps) {
        for (let i = 1; i <= 4; i++) {
            const stepDot = document.getElementById(`step-${i}`);
            if (stepDot) {
                stepDot.classList.remove('active', 'completed');
                if (i <= totalSteps) {
                    if (i === activeStep + 1) {
                        stepDot.classList.add('active');
                    } else if (i < activeStep + 1) {
                        stepDot.classList.add('completed');
                    }
                }
            }
        }
    }

    // Function to stop thought process
    function stopThoughtProcess() {
        if (thoughtProcessInterval) {
            clearTimeout(thoughtProcessInterval);
            thoughtProcessInterval = null;
        }

        if (thoughtProcessContainer) {
            thoughtProcessContainer.style.display = 'none';
        }

        // Reset step indicators
        for (let i = 1; i <= 4; i++) {
            const stepDot = document.getElementById(`step-${i}`);
            if (stepDot) {
                stepDot.classList.remove('active', 'completed');
            }
        }

        currentThoughtStep = 0;
    }

    // ======================================
    // RESIZE FUNCTIONALITY
    // ======================================

    // Initialize resize functionality
    if (resizeHandle) {
        resizeHandle.addEventListener('mousedown', initResize);
    }

    function initResize(e) {
        e.preventDefault();
        isResizing = true;
        startX = e.clientX;
        startY = e.clientY;
        startWidth = parseInt(window.getComputedStyle(chatbotContainer).width, 10);
        startHeight = parseInt(window.getComputedStyle(chatbotContainer).maxHeight, 10);

        // Add global event listeners
        document.addEventListener('mousemove', doResize);
        document.addEventListener('mouseup', stopResize);

        // Prevent text selection during resize
        document.body.style.userSelect = 'none';
        chatbotContainer.style.transition = 'none'; // Disable transitions during resize
    }

    function doResize(e) {
        if (!isResizing) return;

        // For top-left resize, we need to invert the delta calculations
        const deltaX = startX - e.clientX; // Inverted: moving left increases width
        const deltaY = startY - e.clientY; // Inverted: moving up increases height

        const newWidth = startWidth + deltaX;
        const newHeight = startHeight + deltaY;

        // Apply constraints
        const minWidth = 300;
        const maxWidth = Math.min(600, window.innerWidth - 40);
        const minHeight = 400;
        const maxHeight = Math.min(window.innerHeight * 0.8, window.innerHeight - 40);

        const constrainedWidth = Math.max(minWidth, Math.min(maxWidth, newWidth));
        const constrainedHeight = Math.max(minHeight, Math.min(maxHeight, newHeight));

        // Update container size
        chatbotContainer.style.width = constrainedWidth + 'px';
        chatbotContainer.style.maxHeight = constrainedHeight + 'px';

        // Update window height if chatbot is open
        if (isOpen && chatbotWindow) {
            chatbotWindow.style.height = (constrainedHeight - 50) + 'px'; // Subtract header height
        }
    }

    function stopResize() {
        if (!isResizing) return;

        isResizing = false;

        // Remove global event listeners
        document.removeEventListener('mousemove', doResize);
        document.removeEventListener('mouseup', stopResize);

        // Restore text selection and transitions
        document.body.style.userSelect = 'auto';
        chatbotContainer.style.transition = ''; // Re-enable transitions
    }

    // Update window height when chatbot is toggled
    function updateChatbotWindowHeight() {
        if (isOpen && chatbotWindow) {
            const containerHeight = parseInt(window.getComputedStyle(chatbotContainer).maxHeight, 10);
            chatbotWindow.style.height = (containerHeight - 50) + 'px';
        }
    }

    // ======================================
    // EXISTING FUNCTIONALITY
    // ======================================

    // Toggle chatbot window
    chatbotToggle.addEventListener('click', function() {
        isOpen = !isOpen;
        if (isOpen) {
            chatbotWindow.classList.add('show');
            chatbotContainer.classList.add('expanded');
            chatbotToggle.innerHTML = '<i class="fas fa-times"></i>';
            updateChatbotWindowHeight(); // Update height when opening
        } else {
            chatbotWindow.classList.remove('show');
            chatbotContainer.classList.remove('expanded');
            chatbotToggle.innerHTML = '<i class="fas fa-comments"></i>';
        }

        // No need to check status here since we check on page load
    });

    // Tools dropdown functionality
    if (toolsBtn && toolsMenu) {
        // Toggle tools menu
        toolsBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            const isVisible = toolsMenu.style.display === 'block';
            toolsMenu.style.display = isVisible ? 'none' : 'block';
        });

        // Close menu when clicking outside
        document.addEventListener('click', function(e) {
            if (!toolsBtn.contains(e.target) && !toolsMenu.contains(e.target)) {
                toolsMenu.style.display = 'none';
            }
        });

        // Close menu after selecting an option
        toolsMenu.querySelectorAll('.menu-option').forEach(option => {
            option.addEventListener('click', function() {
                setTimeout(() => {
                    toolsMenu.style.display = 'none';
                }, 100);
            });
        });
    }

    // Character counter
    chatbotInput.addEventListener('input', function() {
        const length = this.value.length;
        charCounter.textContent = `${length}/500`;
        charCounter.style.color = length > 450 ? '#ff4444' : '#666';
    });

    // Send message on Enter
    chatbotInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            e.preventDefault(); // Prevent default form submission
            sendMessageWithStreaming();
        }
    });

    // Send message on button click
    chatbotSend.addEventListener('click', function(e) {
        e.preventDefault(); // Prevent any default behavior
        sendMessageWithStreaming();
    });

    // Export dropdown functionality - REMOVED (integrated into tools dropdown)
    // The export functionality is now part of the unified tools dropdown

    // File upload functionality (triggered by tools dropdown)
    document.getElementById('image-upload').addEventListener('change', function(e) {
        if (e.target.files && e.target.files.length > 0) {
            stageImage(e.target.files[0]);  // Stage instead of immediate send
            // Clear the file input after staging
            e.target.value = '';
        }
    });

    document.getElementById('pdf-upload').addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            uploadPDFDocument(e.target.files[0]);
            // Clear the file input after processing
            e.target.value = '';
        }
    });

    // Check AI system status
    function checkAIStatus() {
        if (statusCheckInProgress) return;

        statusCheckInProgress = true;
        // Get fresh references to DOM elements
        const statusText = document.getElementById('ai-status-text');
        const statusIndicator = document.getElementById('ai-status-indicator');

        if (statusText) {
            statusText.textContent = 'Checking...';
        }
        if (statusIndicator) {
            statusIndicator.className = 'status-indicator checking';
        }

        fetch('/api/ai-status')
            .then(response => {
                return response.json();
            })
            .then(data => {
                aiStatus = data;
                updateAIStatus(data);
                statusCheckInProgress = false;
            })
            .catch(error => {
                console.error('AI Status check failed:', error);
                updateAIStatus({
                    ai_api_available: false,
                    local_rag_available: false,
                    architecture: 'error',
                    error: 'Connection failed'
                });
                statusCheckInProgress = false;
            });
    }

    // Update AI status display
    function updateAIStatus(status) {
        // Get fresh references to DOM elements
        const statusIndicator = document.getElementById('ai-status-indicator');
        const statusText = document.getElementById('ai-status-text');
        const statusMessage = document.getElementById('ai-status-message');

        // Check if elements exist
        if (!statusIndicator || !statusText) {
            return;
        }

        // Include ollama_available in the health check
        const isHealthy = status.ai_api_available || status.local_rag_available || status.ollama_available;

        if (isHealthy) {
            statusIndicator.className = 'status-indicator online';
            statusText.textContent = 'Online';
            if (statusMessage) {
                statusMessage.innerHTML = '✅ <strong>AI system ready!</strong> What would you like to know?';
                statusMessage.style.color = '#28a745';
            }
        } else if (status.error) {
            statusIndicator.className = 'status-indicator error';
            statusText.textContent = 'Error';
            if (statusMessage) {
                statusMessage.innerHTML = '❌ <strong>AI system error.</strong> Please try again later or contact support.';
                statusMessage.style.color = '#dc3545';
            }
        } else {
            statusIndicator.className = 'status-indicator offline';
            statusText.textContent = 'Offline';
            if (statusMessage) {
                statusMessage.innerHTML = '⚠️ <strong>AI system offline.</strong> Some features may be limited.';
                statusMessage.style.color = '#ffc107';
            }
        }

        // Update welcome message architecture
        const welcomeArchitecture = document.getElementById('welcome-architecture');
        if (welcomeArchitecture) {
            welcomeArchitecture.textContent = getArchitectureBadge(status.architecture);
            welcomeArchitecture.className = `architecture-badge ${status.architecture}`;
        }
    }

    // Feature detection for streaming support
    let STREAMING_FEATURES = {
        SSE_SUPPORTED: typeof(EventSource) !== "undefined",
        STREAMING_ENABLED: false, // Will be detected from backend
        FALLBACK_AVAILABLE: true,
        MAX_TIMEOUT: 60000
    };

    // Check backend streaming capabilities
    function checkStreamingCapabilities() {
        fetch('/api/features')
            .then(response => response.json())
            .then(data => {
                if (data.success && data.features) {
                    STREAMING_FEATURES.STREAMING_ENABLED = data.features.streaming_enabled;
                    STREAMING_FEATURES.MAX_TIMEOUT = (data.features.max_streaming_timeout || 60) * 1000;
                } else {
                    STREAMING_FEATURES.STREAMING_ENABLED = false;
                }
            })
            .catch(error => {
                STREAMING_FEATURES.STREAMING_ENABLED = false;
            });
    }

    // Initialize streaming capabilities check
    checkStreamingCapabilities();

    // Send message with streaming support
    function sendMessageWithStreaming() {
        const message = chatbotInput.value.trim();

        // Check if we have staged image OR text message
        if (!message && !stagedImage) return;

        // If image is staged, send via image endpoint with optional text
        if (stagedImage) {
            sendImageMessage(stagedImage, message);
            return;
        }

        // Otherwise, normal text message flow
        // Add user message
        addMessage(message, 'user');

        // Clear input
        chatbotInput.value = '';
        charCounter.textContent = '0/500';

        // Start real-time streaming indicator
        startStreamingIndicator();

        // Get document filter value
        const documentFilter = document.getElementById('document-filter');
        const filterValue = documentFilter ? documentFilter.value : 'all';

        // Prepare request data
        const requestData = {
            message: message,
            mode: window.currentMode,
            use_agent: window.currentMode !== 'sop',
            force_rag: window.currentMode === 'sop',
            document_filter: filterValue
        };

        // Use Server-Sent Events for real-time updates
        if (STREAMING_FEATURES.SSE_SUPPORTED && STREAMING_FEATURES.STREAMING_ENABLED) {
            sendWithSSE(requestData);
        } else {
            // Fallback to regular API
            sendMessage();
        }
    }

    // Server-Sent Events implementation
    function sendWithSSE(requestData) {
        // Create EventSource for streaming
        const eventSource = new EventSource('/api/chatbot-stream?' + new URLSearchParams({
            message: requestData.message,
            mode: requestData.mode,
            use_agent: requestData.use_agent,
            force_rag: requestData.force_rag,
            document_filter: requestData.document_filter
        }));

        let finalResponse = null;
        let sessionId = null;

        eventSource.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);

                if (data.session_id) {
                    sessionId = data.session_id;
                }

                switch(data.type) {
                    case 'status':
                        updateStreamingStatus(data.message, data.progress, data.step);
                        break;

                    case 'keepalive':
                        // Just keep the connection alive, no UI update needed
                        break;

                    case 'response':
                        finalResponse = data.data;
                        eventSource.close();
                        stopStreamingIndicator();

                        // Add the final response
                        addMessage(finalResponse.response, 'ai', {
                            confidence: finalResponse.confidence,
                            architecture: finalResponse.architecture,
                            processing_time: finalResponse.processing_time,
                            session_id: finalResponse.session_id,
                            mode: finalResponse.mode || window.currentMode,
                            citations: finalResponse.citations || [],
                            timestamp: new Date()
                        });

                        // If backend provided tools used, append expandable pills under the AI message
                        try {
                            if (Array.isArray(finalResponse.tools_used) && finalResponse.tools_used.length) {
                                appendToolsUsed(finalResponse.tools_used);
                            }
                        } catch (e) { /* Ignore tools_used render errors */ }
                        break;

                    case 'error':
                        eventSource.close();
                        stopStreamingIndicator();
                        addMessage(`Sorry, I encountered an error: ${data.message}`, 'ai', {
                            confidence: 0.0,
                            architecture: 'error',
                            timestamp: new Date()
                        });
                        break;
                }
            } catch (e) {
                console.error('SSE parsing error:', e);
                eventSource.close();
                stopStreamingIndicator();
                // Fallback to regular API
                sendMessage();
            }
        };

        eventSource.onerror = function(event) {
            console.error('SSE connection error:', event);
            eventSource.close();
            stopStreamingIndicator();

            // Fallback to regular API call
            sendMessage();
        };

        // Extended timeout for complex queries - 120 seconds to match backend
        const timeoutId = setTimeout(() => {
            if (eventSource.readyState !== EventSource.CLOSED) {
                eventSource.close();
                stopStreamingIndicator();
                addMessage('Connection lost. Please try again.', 'ai', {
                    confidence: 0.0,
                    architecture: 'timeout',
                    timestamp: new Date()
                });
            }
        }, 120000);

        // Clear timeout when connection closes normally
        eventSource.addEventListener('close', () => {
            clearTimeout(timeoutId);
        });
    }

    // Enhanced streaming status indicator
    function startStreamingIndicator() {
        // Force remove any old streaming indicators first
        stopStreamingIndicator(true);

        // Hide the old thought process container
        if (thoughtProcessContainer) {
            thoughtProcessContainer.style.display = 'none';
        }

        // Create new streaming status element
        let streamingStatus = document.createElement('div');
        streamingStatus.id = 'streaming-status';
        streamingStatus.className = 'streaming-status';
        streamingStatus.innerHTML = `
                <div class="streaming-content">
                    <i class="fas fa-brain fa-pulse streaming-icon"></i>
                    <span id="streaming-text">Initializing AI processing...</span>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress-fill"></div>
                    </div>
                    <div class="streaming-details" id="streaming-details">
                        <span class="step-indicator" id="step-indicator">Starting...</span>
                    </div>
                </div>
        `;
        chatbotMessages.appendChild(streamingStatus);
        streamingStatus.style.display = 'block';
        typingIndicator.style.display = 'block';
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    }

    function updateStreamingStatus(message, progress, step) {
        const streamingText = document.getElementById('streaming-text');
        const progressFill = document.getElementById('progress-fill');
        const stepIndicator = document.getElementById('step-indicator');

        if (streamingText) {
            streamingText.textContent = message;
        }

        if (progressFill) {
            progressFill.style.width = `${progress}%`;
            progressFill.style.backgroundColor = progress > 80 ? '#28a745' : progress > 40 ? '#ffc107' : '#007bff';
        }

        if (stepIndicator && step) {
            stepIndicator.textContent = step.replace('_', ' ').toUpperCase();
            stepIndicator.className = `step-indicator step-${step}`;
        }

        // Auto-scroll to keep status visible
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    }

    function stopStreamingIndicator(force = false) {
        const streamingStatus = document.getElementById('streaming-status');

        if (force && streamingStatus) {
            // Force remove - completely delete the old indicator
            streamingStatus.remove();
        } else if (streamingStatus) {
            // Normal hide
            streamingStatus.style.display = 'none';
        }

        typingIndicator.style.display = 'none';

        // Stop the old thought process as well
        stopThoughtProcess();
    }

    // Original send message function (fallback)
    function sendMessage() {
        const message = chatbotInput.value.trim();
        if (!message) return;

        // Add user message
        addMessage(message, 'user');

        // Clear input
        chatbotInput.value = '';
        charCounter.textContent = '0/500';

        // Start thought process instead of simple typing indicator
        startThoughtProcess(window.currentMode);
        typingIndicator.style.display = 'block';

        // Prepare request based on current mode
        const requestData = {
            message: message,
            mode: window.currentMode,
            use_agent: window.currentMode !== 'sop', // Use agent only in chat mode
            force_rag: window.currentMode === 'sop'  // Force RAG in SOP mode
        };

        // Send to API with fallback for testing
        fetch('/api/chatbot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            stopThoughtProcess();
            typingIndicator.style.display = 'none';

            if (data.success) {
                addMessage(data.response, 'ai', {
                    confidence: data.confidence,
                    architecture: data.architecture,
                    reasoning: data.reasoning,
                    mode: data.mode || window.currentMode,
                    citations: data.citations || [],
                    timestamp: new Date()
                });
            } else {
                addMessage('Sorry, I encountered an error: ' + data.error, 'ai', {
                    confidence: 0.0,
                    architecture: 'error',
                    mode: window.currentMode,
                    timestamp: new Date()
                });
            }
        })
        .catch(error => {
            stopThoughtProcess();
            typingIndicator.style.display = 'none';
            console.error('Chatbot API error:', error);

            // FALLBACK: Provide a test response based on mode
            let fallbackResponse = '';
            let fallbackArchitecture = 'fallback_test';

            switch (window.currentMode) {
                case 'chat':
                    fallbackResponse = ` **Chat Mode Test Response**\n\nYou asked: "${message}"\n\nThis is a fallback response since the API server isn't available. In normal operation, I would provide a conversational AI response here.\n\n🔧 **Status**: API server not running - please start with \`py api.py\``;
                    break;
                case 'sop':
                    fallbackResponse = ` **SOP Mode Test Response**\n\nYou searched for: "${message}"\n\nThis is a fallback response since the API server isn't available. In normal operation, I would search through QKD documentation and procedures to answer your question.\n\n🔧 **Status**: API server not running - please start with \`py api.py\``;
                    break;
                case 'agent':
                    fallbackResponse = ` **Agent Mode Test Response**\n\nYou requested: "${message}"\n\nThis is a fallback response since the API server isn't available. In normal operation, I would use AI tools and reasoning to solve complex problems.\n\n🔧 **Status**: API server not running - please start with \`py api.py\``;
                    break;
                default:
                    fallbackResponse = ` **Connection Error**\n\nUnable to connect to the AI API server. The chat interface is working, but AI responses are not available.\n\n🔧 **To Fix**: Start the API server with \`py api.py\``;
            }

            addMessage(fallbackResponse, 'ai', {
                confidence: 0.1,
                architecture: fallbackArchitecture,
                mode: window.currentMode,
                timestamp: new Date()
            });
        });
    }

    // Staged Image Functions
    function stageImage(imageFile) {
        // Validate file
        if (!imageFile) {
            return;
        }

        // Check file type
        if (!imageFile.type.startsWith('image/')) {
            addErrorMessage('Please select a valid image file');
            return;
        }

        // Check file size (10MB limit)
        const maxSize = 10 * 1024 * 1024; // 10MB
        if (imageFile.size > maxSize) {
            addErrorMessage('Image file size must be less than 10MB');
            return;
        }

        // Store the image
        stagedImage = imageFile;

        // Show preview
        const previewContainer = document.getElementById('staged-image-preview');
        const thumbnail = document.getElementById('staged-image-thumbnail');

        const reader = new FileReader();
        reader.onload = function(e) {
            thumbnail.src = e.target.result;
            previewContainer.style.display = 'block';
        };
        reader.onerror = function(e) {
            addErrorMessage('Failed to load image preview');
        };
        reader.readAsDataURL(imageFile);

        // Focus input so user can type question
        chatbotInput.focus();
    }

    function clearStagedImage() {
        stagedImage = null;
        const previewContainer = document.getElementById('staged-image-preview');
        const thumbnail = document.getElementById('staged-image-thumbnail');
        if (previewContainer) previewContainer.style.display = 'none';
        if (thumbnail) thumbnail.src = '';

        // Also clear the file input
        const fileInput = document.getElementById('image-upload');
        if (fileInput) fileInput.value = '';
    }

    // Add remove button handler
    document.getElementById('remove-staged-image').addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation(); // Prevent event bubbling
        clearStagedImage();
    });

    // Enhanced Image Upload Handling (renamed internally, called with staged image)
    function sendImageMessage(imageFile, textMessage = '') {
        // Validate file
        if (!imageFile) {
            return;
        }

        // Check file type (validation already done in stageImage, but double-check)
        if (!imageFile.type.startsWith('image/')) {
            addErrorMessage('Please select a valid image file');
            clearStagedImage();
            return;
        }

        // Check file size (10MB limit)
        const maxSize = 10 * 1024 * 1024; // 10MB
        if (imageFile.size > maxSize) {
            addErrorMessage('Image file size must be less than 10MB');
            clearStagedImage();
            return;
        }

        const formData = new FormData();
        formData.append('image', imageFile);

        // Add optional text message
        if (textMessage && textMessage.trim()) {
            formData.append('message', textMessage.trim());
        }

        // Show image preview in chat
        addImageMessage(imageFile, textMessage);

        // Clear staged image and input
        clearStagedImage();
        chatbotInput.value = '';
        charCounter.textContent = '0/500';

        // Show typing indicator and thought process
        startThoughtProcess('chat'); // Image uploads use chat mode
        typingIndicator.style.display = 'block';

        // Send to API
        fetch('/api/chatbot', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            stopThoughtProcess();
            typingIndicator.style.display = 'none';

            if (data.success) {
                // Check architecture type and use appropriate message function
                if (data.architecture === 'enhanced_workflow' || data.architecture === 'visual_search_enhanced') {
                    // Use enhanced visual search message with device label info
                    addEnhancedWorkflowMessage(data.response, data.workflow_results, data.confidence);
                } else {
                    // Fall back to basic image detection message
                    addAIMessage(data.response, data.detected_equipment, data.confidence);
                }
            } else {
                addErrorMessage(data.error);
            }
        })
        .catch(error => {
            stopThoughtProcess();
            typingIndicator.style.display = 'none';
            addErrorMessage('Failed to process image');
        });
    }

    function addImageMessage(imageFile, textMessage) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message';

        const timestamp = new Date();
        const timeString = timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        // Create message structure
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        const messageTextDiv = document.createElement('div');
        messageTextDiv.className = 'message-text';

        // Create image element
        const img = document.createElement('img');
        img.style.maxWidth = '300px';
        img.style.maxHeight = '300px';
        img.style.display = 'block';
        img.style.borderRadius = '8px';

        // Use FileReader for more reliable image loading
        if (imageFile && (imageFile instanceof File || imageFile instanceof Blob)) {
            const reader = new FileReader();

            reader.onload = function(e) {
                img.src = e.target.result;
            };

            reader.onerror = function(e) {
                img.alt = 'Failed to load image';
            };

            // Read the file as data URL (base64)
            reader.readAsDataURL(imageFile);
        } else {
            img.alt = 'Invalid image file';
        }

        // Add image to message
        messageTextDiv.appendChild(img);

        // Only add text message if provided
        if (textMessage) {
            const textDiv = document.createElement('div');
            textDiv.style.marginTop = '10px';
            textDiv.textContent = textMessage;
            messageTextDiv.appendChild(textDiv);
        }

        // Add timestamp
        const metaDiv = document.createElement('div');
        metaDiv.className = 'message-meta';
        metaDiv.innerHTML = `<span class="timestamp">${timeString}</span>`;

        // Assemble and append
        messageContent.appendChild(messageTextDiv);
        messageContent.appendChild(metaDiv);
        messageDiv.appendChild(messageContent);

        chatbotMessages.appendChild(messageDiv);
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    }

    // PDF Upload functionality
    async function uploadPDFDocument(pdfFile) {
        try {
            // Validate file size (50MB limit)
            const maxSize = 50 * 1024 * 1024; // 50MB
            if (pdfFile.size > maxSize) {
                addMessage(' File too large. Maximum size is 50MB.', 'system');
                return;
            }

            // No need for vendor detection - let backend handle it automatically

            // Show upload message
            addMessage(` Uploading ${pdfFile.name}...`, 'system');

            // Prepare form data
            const formData = new FormData();
            formData.append('file', pdfFile);

            // Upload file
            const uploadResponse = await fetch('/api/upload_pdf', {
                method: 'POST',
                body: formData
            });

            const uploadData = await uploadResponse.json();

            if (!uploadData.success) {
                throw new Error(uploadData.error || 'Upload failed');
            }

            const jobId = uploadData.job_id;
            addMessage(` Processing ${pdfFile.name}. This may take a few minutes...`, 'system');

            // Poll for processing status
            pollUploadStatus(jobId, pdfFile.name);

        } catch (error) {
            console.error('PDF upload error:', error);
            addMessage(` Upload failed: ${error.message}`, 'system');
        } finally {
            // Clear file input
            document.getElementById('pdf-upload').value = '';
        }
    }

    async function pollUploadStatus(jobId, filename) {
        const maxPolls = 240; // 20 minutes max (5s intervals)
        let pollCount = 0;

        const poll = async () => {
            try {
                pollCount++;

                const response = await fetch(`/api/upload_status/${jobId}`);
                const data = await response.json();

                if (!data.success) {
                    throw new Error(data.error || 'Status check failed');
                }

                if (data.status === 'processing') {
                    // Update progress message if available
                    if (data.message && pollCount % 6 === 0) { // Every 30 seconds
                        addMessage(` ${data.message}`, 'system');
                    }

                    if (pollCount < maxPolls) {
                        setTimeout(poll, 5000); // Poll every 5 seconds
                    } else {
                        addMessage(` Processing ${filename} is taking longer than expected. It will continue in the background.`, 'system');
                    }

                } else if (data.status === 'completed') {
                    const result = data.result || {};
                    const chunks = result.chunks || 0;
                    const displayName = result.display_name || filename;

                    addMessage(` Successfully indexed ${filename} (${chunks} chunks). You can now query this document!`, 'system');

                    // Rebuild indexes so the new document is searchable
                    rebuildIndexesInBackground();

                    // Update document filter dropdown
                    updateDocumentFilter();

                } else if (data.status === 'failed') {
                    const error = data.error || 'Processing failed';
                    addMessage(` Failed to process ${filename}: ${error}`, 'system');

                } else if (data.status === 'queued') {
                    if (pollCount < maxPolls) {
                        setTimeout(poll, 5000);
                    }
                }

            } catch (error) {
                console.error('Status polling error:', error);
                addMessage(` Failed to check processing status: ${error.message}`, 'system');
            }
        };

        // Start polling
        setTimeout(poll, 2000); // Initial delay
    }

    // Make updateDocumentFilter globally accessible
    window.updateDocumentFilter = async function() {
        try {
            const response = await fetch('/api/documents');

            if (!response.ok) {
                return;
            }

            const data = await response.json();

            if (!data.success) {
                return;
            }

            const documentFilter = document.getElementById('document-filter');
            if (!documentFilter) return;

            // Clear existing dynamic options (keep only 'all')
            const options = documentFilter.querySelectorAll('option');
            options.forEach(option => {
                if (option.value !== 'all') {
                    option.remove();
                }
            });

            // Add each document as a separate option
            data.documents.forEach(doc => {
                // Create option with the actual filename as value
                const option = document.createElement('option');
                option.value = doc.filename;  // Use actual filename as value (preserve case)
                option.textContent = doc.display_name || doc.filename.replace('.pdf', '').replace('.pdf', ''); // Remove double .pdf
                option.dataset.filename = doc.filename;
                documentFilter.appendChild(option);
            });

        } catch (error) {
            // Silently fail - document filter is non-critical
        }
    }

    // Convert markdown-like text to HTML while preserving formatting
    function formatMessageText(text) {
        // Preserve the structure and formatting
        let formatted = text;

        // Convert markdown bold **text** to <strong>
        formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Convert markdown code `text` to <code>
        formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Convert line breaks to <br> tags
        formatted = formatted.replace(/\n/g, '<br>');

        // Preserve indentation by converting spaces to non-breaking spaces
        // Match lines that start with spaces (for indented items)
        formatted = formatted.replace(/(<br>)([ ]+)/g, function(match, br, spaces) {
            return br + spaces.replace(/ /g, '&nbsp;');
        });

        // Also handle spaces at the beginning of the text
        formatted = formatted.replace(/^([ ]+)/g, function(match, spaces) {
            return spaces.replace(/ /g, '&nbsp;');
        });

        // Convert lists with proper structure
        // Handle numbered lists (1. 2. etc)
        formatted = formatted.replace(/(<br>|^)(\d+)\.\s+/g, '$1<span class="list-number">$2.</span>&nbsp;');

        // Handle bullet points with dashes
        formatted = formatted.replace(/(<br>|^)(&nbsp;)*-\s+/g, '$1$2•&nbsp;');

        // Wrap the whole thing in a div to preserve block formatting
        return `<div class="formatted-message">${formatted}</div>`;
    }

    // Add message to chat
    function addMessage(text, type, metadata = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;

        if (type === 'user') {
            window.lastUserMessage = text;
        }

        const timestamp = metadata.timestamp || new Date();
        const timeString = timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        // Format the text to preserve structure
        const formattedText = formatMessageText(text);

        let messageHTML = `
            <div class="message-content">
                <div class="message-text">${formattedText}</div>
                <div class="message-meta">
                    <span class="timestamp">${timeString}</span>
        `;

        if (type === 'ai') {
            // Add confidence bar
            const confidence = metadata.confidence || 0;
            const confidencePercent = Math.round(confidence * 100);
            const confidenceClass = confidence > 0.7 ? 'high' : confidence > 0.4 ? 'medium' : 'low';

            messageHTML += `
                <span class="confidence-container">
                    <span class="confidence-label">Confidence:</span>
                    <span class="confidence-bar">
                        <span class="confidence-fill ${confidenceClass}" style="width: ${confidencePercent}%"></span>
                    </span>
                    <span class="confidence-text">${confidencePercent}%</span>
                </span>
            `;

            // Add architecture badge
            const architecture = metadata.architecture || 'unknown';
            messageHTML += `
                <span class="architecture-badge ${architecture}">${getArchitectureBadge(architecture)}</span>
            `;
        }

        messageHTML += `
                </div>
            </div>
        `;

        messageDiv.innerHTML = messageHTML;

        // Add reasoning section if available
        if (type === 'ai' && metadata.reasoning && metadata.reasoning.steps > 0) {
            const reasoningDiv = document.createElement('div');
            reasoningDiv.className = 'reasoning-section';
            reasoningDiv.innerHTML = createReasoningHTML(metadata.reasoning);
            messageDiv.appendChild(reasoningDiv);
        }

        // Append suggestion chips under AI replies
        if (type === 'ai') {
            const responseMode = metadata.mode || window.currentMode || 'chat';
            const architecture = (metadata.architecture || '').toLowerCase();
            const isError = architecture === 'error' || architecture === 'timeout';
            const chips = isError ? [] : getSuggestionChips(responseMode);
            if (chips && chips.length) {
                messageDiv.appendChild(createSuggestionChips(chips));
            }
            // Citations panel for SOP mode
            const citations = Array.isArray(metadata.citations) ? metadata.citations : [];
            if (responseMode === 'sop') {
                let effectiveCitations = citations;
                if (!effectiveCitations || effectiveCitations.length === 0) {
                    // Fallback: parse from raw text body
                    effectiveCitations = parseCitationsFromText(text);
                }
                if (effectiveCitations && effectiveCitations.length > 0) {
                    const panel = createCitationsPanel(effectiveCitations);
                    messageDiv.appendChild(panel);
                }
            }
        }

        chatbotMessages.appendChild(messageDiv);
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    }

    // Append expandable Tools Used below last AI message
    function appendToolsUsed(tools) {
        const messages = document.querySelectorAll('#chatbot-messages .message.ai-message');
        if (!messages.length) return;
        const last = messages[messages.length - 1];
        const wrapper = document.createElement('div');
        wrapper.className = 'tools-used-wrapper';
        const id = 'tools-used-' + Date.now();
        const pills = tools.map(t => `<span class="suggestion-chip" style="cursor:default">${t}</span>`).join('');
        wrapper.innerHTML = `
            <div class="tools-used-bar" role="button" aria-expanded="false" aria-controls="${id}">
                <i class="fas fa-wrench" aria-hidden="true"></i>
                <span class="tools-used-title">Tools used • ${tools.length}</span>
                <i class="fas fa-chevron-down tools-used-caret" aria-hidden="true"></i>
            </div>
            <div id="${id}" class="tools-used-content" style="display:none">
                <div class="tools-used-pills">${pills}</div>
            </div>`;
        last.appendChild(wrapper);
        const bar = wrapper.querySelector('.tools-used-bar');
        const content = wrapper.querySelector('.tools-used-content');
        const caret = wrapper.querySelector('.tools-used-caret');
        bar.addEventListener('click', () => {
            const expanded = content.style.display !== 'none';
            content.style.display = expanded ? 'none' : 'block';
            bar.setAttribute('aria-expanded', String(!expanded));
            caret.className = expanded ? 'fas fa-chevron-down tools-used-caret' : 'fas fa-chevron-up tools-used-caret';
        });
    }

    // Create reasoning HTML
    function createReasoningHTML(reasoning) {
        let html = `
            <div class="reasoning-header">
                <i class="fas fa-brain"></i>
                <span>AI Reasoning Process</span>
                <button class="reasoning-toggle" onclick="toggleReasoning(this)">
                    <i class="fas fa-chevron-down"></i>
                </button>
            </div>
            <div class="reasoning-content">
                <div class="reasoning-stats">
                    <span><strong>Steps:</strong> ${reasoning.steps}</span>
                    <span><strong>Processing Time:</strong> ${reasoning.processing_time?.toFixed(2) || 0}s</span>
                    ${reasoning.tools_used?.length ? `<span><strong>Tools Used:</strong> ${reasoning.tools_used.join(', ')}</span>` : ''}
                </div>
        `;

        if (reasoning.detailed_steps && reasoning.detailed_steps.length > 0) {
            html += '<div class="reasoning-steps">';
            reasoning.detailed_steps.forEach((step, index) => {
                html += `
                    <div class="reasoning-step">
                        <div class="step-header">
                            <span class="step-number">${index + 1}</span>
                            <span class="step-action">${step.action || 'Unknown Action'}</span>
                        </div>
                        ${step.thought ? `<div class="step-thought">💭 <strong>Thought:</strong> ${step.thought}</div>` : ''}
                        ${step.action_input ? `<div class="step-input">📝 <strong>Input:</strong> <code>${step.action_input}</code></div>` : ''}
                        ${step.observation ? `<div class="step-observation">👁️ <strong>Observation:</strong> ${step.observation}</div>` : ''}
                    </div>
                `;
            });
            html += '</div>';
        }

        html += '</div>';
        return html;
    }

    // Add AI message specifically for image detection responses
    function addAIMessage(response, detectedEquipment, confidence) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message ai-message';

        const timestamp = new Date();
        const timeString = timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const confidencePercent = Math.round(confidence * 100);
        const confidenceClass = confidence > 0.7 ? 'high' : confidence > 0.4 ? 'medium' : 'low';

        // Format the response text to preserve structure
        const formattedResponse = formatMessageText(response);

        let messageHTML = `
            <div class="message-content">
                <div class="message-text">${formattedResponse}</div>
                <div class="message-meta">
                    <span class="timestamp">${timeString}</span>
                    <span class="confidence-container">
                        <span class="confidence-label">Confidence:</span>
                        <span class="confidence-bar">
                            <span class="confidence-fill ${confidenceClass}" style="width: ${confidencePercent}%"></span>
                        </span>
                        <span class="confidence-text">${confidencePercent}%</span>
                    </span>
                    <span class="architecture-badge image_detection">CV Detection</span>
                </div>
            </div>
        `;

        messageDiv.innerHTML = messageHTML;
        chatbotMessages.appendChild(messageDiv);
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    }

    function addErrorMessage(error) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message ai-message';

        const timestamp = new Date();
        const timeString = timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-text"> ${error}</div>
                <div class="message-meta">
                    <span class="timestamp">${timeString}</span>
                    <span class="architecture-badge error">Error</span>
                </div>
            </div>
        `;

        chatbotMessages.appendChild(messageDiv);
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    }

    // Suggestion chip helpers
    function getSuggestionChips(mode) {
        switch ((mode || '').toLowerCase()) {
            case 'sop':
                return [
                    'Summarize the steps as a checklist.',
                    'Show the sources you used.',
                    'Find related SOPs or troubleshooting tips.'
                ];
            case 'agent':
                return [
                    'Propose next actions and owners.',
                    'Run a quick diagnosis.',
                    'What data or access do you need to proceed?'
                ];
            case 'chat':
            default:
                return [
                    'Summarize in 3 bullet points.',
                    'Give a concrete example.',
                    'What are the risks and caveats?'
                ];
        }
    }

    function createSuggestionChips(chips) {
        const container = document.createElement('div');
        container.className = 'suggestion-chips';
        container.setAttribute('role', 'group');
        container.setAttribute('aria-label', 'Suggested follow-ups');
        chips.forEach(text => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'suggestion-chip';
            btn.textContent = text;
            btn.addEventListener('click', () => {
                chatbotInput.value = text;
                chatbotInput.focus();
                if (typeof sendMessageWithStreaming === 'function') {
                    sendMessageWithStreaming();
                } else {
                    document.getElementById('chatbot-send').click();
                }
            });
            container.appendChild(btn);
        });
        return container;
    }

    // Citations panel helpers
    function createCitationsPanel(citations) {
        const wrapper = document.createElement('div');
        wrapper.className = 'citations-panel';
        wrapper.style.marginTop = '8px';
        const label = document.createElement('div');
        label.textContent = 'Sources:';
        label.style.cssText = 'font-size:12px;color:#555;margin-bottom:6px;font-weight:600;';
        wrapper.appendChild(label);

        const row = document.createElement('div');
        row.className = 'citations-row';
        row.style.cssText = 'display:flex;gap:8px;flex-wrap:wrap;';
        citations.forEach(c => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'suggestion-chip';
            const pageText = Array.isArray(c.pages) && c.pages.length ? ` (p. ${c.pages.slice(0,3).join(',')}${c.pages.length>3?'…':''})` : '';
            btn.textContent = `${c.doc || c.source_file || 'Source'}${pageText}`;
            btn.addEventListener('click', (e) => {
                // If URL available, open in new tab at page anchor; else show modal
                if (c.url) {
                    window.open(c.url, '_blank');
                } else {
                    openCitationModal(c);
                }
            });
            row.appendChild(btn);
        });
        wrapper.appendChild(row);
        return wrapper;
    }

    function openCitationModal(citation) {
        let modal = document.getElementById('citation-modal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'citation-modal';
            modal.className = 'ai-modal';
            modal.innerHTML = `
                <div class="ai-modal-content">
                    <div class="ai-modal-header">
                        <div class="ai-modal-title" id="citation-title"></div>
                        <button class="ai-modal-close" id="citation-close">×</button>
                    </div>
                    <div class="ai-modal-body">
                        <div id="citation-meta" style="font-size:12px;color:#666;margin-bottom:8px;"></div>
                        <div id="citation-excerpt" style="white-space:pre-wrap;line-height:1.5;"></div>
                    </div>
                </div>`;
            document.body.appendChild(modal);
            document.getElementById('citation-close').addEventListener('click', closeCitationModal);
            modal.addEventListener('click', (e) => { if (e.target === modal) closeCitationModal(); });
        }
        // Populate content
        const title = document.getElementById('citation-title');
        const meta = document.getElementById('citation-meta');
        const excerpt = document.getElementById('citation-excerpt');
        title.textContent = citation.doc || citation.source_file || 'Source';
        const pages = Array.isArray(citation.pages) && citation.pages.length ? `Pages: ${citation.pages.join(', ')}` : '';
        const sections = Array.isArray(citation.sections) && citation.sections.length ? `Sections: ${citation.sections.join('; ')}` : '';
        meta.textContent = [pages, sections].filter(Boolean).join(' • ');
        excerpt.textContent = citation.excerpt || 'No excerpt available.';
        modal.style.display = 'block';
    }

    function closeCitationModal() {
        const modal = document.getElementById('citation-modal');
        if (modal) modal.style.display = 'none';
    }

    function parseCitationsFromText(raw) {
        try {
            if (!raw || typeof raw !== 'string') return [];
            const citations = [];
            // Normalize
            const text = raw.replace(/\r/g, '');
            // Locate source section markers
            const markerIdx = text.toLowerCase().indexOf('**source:**');
            const marker2 = text.toLowerCase().indexOf('\nsource:');
            const start = markerIdx !== -1 ? markerIdx + '**SOURCE:**'.length : (marker2 !== -1 ? marker2 + 'source:'.length : -1);
            const block = start !== -1 ? text.slice(start) : text;
            const lines = block.split('\n').map(l => l.trim()).filter(Boolean);
            const maxLines = Math.min(lines.length, 6);
            for (let i = 0; i < maxLines; i++) {
                const line = lines[i];
                // Match patterns like: "7.1. Cerberis3 User Guide (v1.24) | Pages 7, 12, 13"
                let m = line.match(/^(.*?)(?:\s*\|\s*Pages?\s*([\d,\s]+))$/i);
                if (!m) {
                    // Alternative: with emoji prefix or 'Page'
                    m = line.match(/^\W*([^|]+?)\s*\|\s*Page(?:s)?\s*([\d,\s]+)$/i);
                }
                if (m) {
                    const doc = m[1].trim().replace(/^\W+/, '');
                    const pagesStr = (m[2] || '').trim();
                    const pages = pagesStr ? pagesStr.split(',').map(p => parseInt(p.trim(), 10)).filter(n => !isNaN(n)) : [];
                    const filename = doc.endsWith('.pdf') ? doc : `${doc}.pdf`;
                    const first = pages.length ? pages[0] : 1;
                    citations.push({
                        doc,
                        source_file: filename,
                        pages,
                        sections: [],
                        excerpt: '',
                        first_page: first,
                        url: `/docs/${encodeURIComponent(filename)}#page=${first}`
                    });
                }
            }
            // Deduplicate by doc
            const seen = new Set();
            const unique = [];
            for (const c of citations) {
                if (!seen.has(c.doc)) { seen.add(c.doc); unique.push(c); }
            }
            return unique;
        } catch (e) {
            return [];
        }
    }

    // Add enhanced workflow message with device label information
    function addEnhancedWorkflowMessage(response, workflowResults, confidence) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message ai-message';

        const timestamp = new Date();
        const timeString = timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const confidencePercent = Math.round(confidence * 100);
        const confidenceClass = confidence > 0.7 ? 'high' : confidence > 0.4 ? 'medium' : 'low';

        // Build device info display
        let deviceInfo = '';
        if (workflowResults && workflowResults.has_device_label && workflowResults.device_label) {
            deviceInfo = `<div class="device-info">
                <strong>🏷️ Device:</strong> ${workflowResults.device_label}
                <span class="detection-confidence">(${workflowResults.detection_confidence})</span>
            </div>`;
        } else if (workflowResults && workflowResults.manufacturer && workflowResults.model) {
            deviceInfo = `<div class="device-info">
                <strong>🏷️ Device:</strong> ${workflowResults.manufacturer} ${workflowResults.model}
            </div>`;
        }

        // Format the response text to preserve structure
        const formattedResponse = formatMessageText(response);

        let messageHTML = `
            <div class="message-content">
                ${deviceInfo}
                <div class="message-text">${formattedResponse}</div>
                <div class="message-meta">
                    <span class="timestamp">${timeString}</span>
                    <span class="confidence-container">
                        <span class="confidence-label">Confidence:</span>
                        <span class="confidence-bar">
                            <span class="confidence-fill ${confidenceClass}" style="width: ${confidencePercent}%"></span>
                        </span>
                        <span class="confidence-text">${confidencePercent}%</span>
                    </span>
                    <span class="architecture-badge enhanced_workflow">Enhanced CV + RAG</span>
                </div>
            </div>
        `;

        messageDiv.innerHTML = messageHTML;
        chatbotMessages.appendChild(messageDiv);
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    }

    // Get architecture badge text
    function getArchitectureBadge(architecture) {
        switch(architecture) {
            case 'agentic_ai': return 'Agent';
            case 'rag_first': return 'RAG';
            case 'local_service': return 'Local';
            case 'mock': return 'Mock';
            case 'error': return 'Error';
            case 'image_detection': return 'CV Detection';
            case 'enhanced_workflow': return 'Enhanced CV + RAG';
            case 'visual_search_enhanced': return 'Visual Search + RAG';
            default: return 'Unknown';
        }
    }

    // Set welcome message timestamp
    const welcomeTimestamp = document.getElementById('welcome-timestamp');
    if (welcomeTimestamp) {
        const now = new Date();
        welcomeTimestamp.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    // Initialize AI status check
    checkAIStatus();

    // Refresh AI status every 30 seconds
    setInterval(checkAIStatus, 30000);

    // ======================================
    // CHAT EXPORT FUNCTIONALITY (NO DEPENDENCIES)
    // ======================================

    // Export chat as PDF file
    function exportChatAsPDF() {
        try {
            const messages = document.querySelectorAll('#chatbot-messages .message');
            if (messages.length === 0) {
                alert('No chat messages to export.');
                return;
            }

            const { jsPDF } = window.jspdf;
            const doc = new jsPDF();

            // Add title
            doc.setFontSize(16);
            doc.setFont(undefined, 'bold');
            doc.text('QKD Chat Conversation Report', 10, 20);

            // Add export date
            doc.setFontSize(10);
            doc.setFont(undefined, 'normal');
            doc.text(`Exported on: ${new Date().toLocaleString()}`, 10, 30);

            // Add separator line
            doc.setLineWidth(0.5);
            doc.line(10, 35, 200, 35);

            let yPosition = 45;
            const pageHeight = doc.internal.pageSize.height;
            const marginBottom = 20;

            messages.forEach((message, index) => {
                const isUser = message.classList.contains('user-message');
                const isAI = message.classList.contains('ai-message');
                const isSystem = message.classList.contains('system-message');

                if (!isUser && !isAI && !isSystem) return;

                const messageTextElement = message.querySelector('.message-text');
                const timestampElement = message.querySelector('.timestamp');

                if (!messageTextElement) return;

                const messageText = messageTextElement.textContent || messageTextElement.innerText || '';
                const timestamp = timestampElement ? timestampElement.textContent : '';

                let sender = 'Unknown';
                if (isUser) {
                    sender = 'User';
                    doc.setTextColor(0, 0, 255);  // Blue for user
                } else if (isAI) {
                    sender = 'AI Assistant';
                    doc.setTextColor(0, 128, 0);  // Green for AI
                } else if (isSystem) {
                    sender = 'System';
                    doc.setTextColor(128, 128, 128);  // Gray for system
                }

                // Check if we need a new page
                if (yPosition > pageHeight - marginBottom) {
                    doc.addPage();
                    yPosition = 20;
                }

                // Add timestamp and sender
                doc.setFontSize(9);
                doc.setFont(undefined, 'bold');
                doc.text(`[${timestamp}] ${sender}:`, 10, yPosition);
                yPosition += 7;

                // Add message text
                doc.setFont(undefined, 'normal');
                doc.setTextColor(0, 0, 0);  // Black for message content
                doc.setFontSize(10);

                // Split long messages into lines
                const splitText = doc.splitTextToSize(messageText.trim(), 180);
                splitText.forEach(line => {
                    if (yPosition > pageHeight - marginBottom) {
                        doc.addPage();
                        yPosition = 20;
                    }
                    doc.text(line, 15, yPosition);
                    yPosition += 5;
                });

                // Add AI metadata if available
                if (isAI) {
                    const confidenceElement = message.querySelector('.confidence-text');
                    const architectureElement = message.querySelector('.architecture-badge');

                    if (confidenceElement || architectureElement) {
                        doc.setFontSize(8);
                        doc.setTextColor(100, 100, 100);
                        let metadata = '';
                        if (confidenceElement) metadata += `Confidence: ${confidenceElement.textContent}`;
                        if (architectureElement) {
                            if (metadata) metadata += ' | ';
                            metadata += `Architecture: ${architectureElement.textContent}`;
                        }
                        doc.text(metadata, 15, yPosition);
                        yPosition += 5;
                    }
                }

                yPosition += 5;  // Add spacing between messages
            });

            // Save the PDF
            doc.save(`chat-report-${getTimestamp()}.pdf`);

        } catch (error) {
            console.error('PDF export error:', error);
            alert('Failed to export chat as PDF. Please try again.');
        }
    }

    // Export chat as HTML file
    function exportChatAsHTML() {
        try {
            const messages = document.querySelectorAll('#chatbot-messages .message');
            if (messages.length === 0) {
                alert('No chat messages to export.');
                return;
            }

            let htmlContent = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat Conversation Export</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        .header { border-bottom: 2px solid #ccc; padding-bottom: 10px; margin-bottom: 20px; }
        .message { margin-bottom: 20px; padding: 10px; border-radius: 8px; }
        .user-message { background-color: #e3f2fd; border-left: 4px solid #2196f3; }
        .ai-message { background-color: #e8f5e8; border-left: 4px solid #4caf50; }
        .system-message { background-color: #fafafa; border-left: 4px solid #999; }
        .message-header { font-weight: bold; margin-bottom: 5px; }
        .message-text { margin-bottom: 8px; }
        .message-meta { font-size: 0.85em; color: #666; }
        .timestamp { margin-right: 10px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Chat Conversation Export</h1>
        <p>Exported on: ${new Date().toLocaleString()}</p>
    </div>
    <div class="messages">`;

            messages.forEach((message, index) => {
                const isUser = message.classList.contains('user-message');
                const isAI = message.classList.contains('ai-message');
                const isSystem = message.classList.contains('system-message');

                if (!isUser && !isAI && !isSystem) return;

                const messageTextElement = message.querySelector('.message-text');
                const timestampElement = message.querySelector('.timestamp');

                if (!messageTextElement) return;

                const messageText = messageTextElement.innerHTML || messageTextElement.textContent || '';
                const timestamp = timestampElement ? timestampElement.textContent : '';

                let sender = 'Unknown';
                let messageClass = '';
                if (isUser) { sender = 'User'; messageClass = 'user-message'; }
                else if (isAI) { sender = 'AI Assistant'; messageClass = 'ai-message'; }
                else if (isSystem) { sender = 'System'; messageClass = 'system-message'; }

                htmlContent += `
        <div class="message ${messageClass}">
            <div class="message-header">${sender}</div>
            <div class="message-text">${messageText}</div>
            <div class="message-meta">
                <span class="timestamp">${timestamp}</span>`;

                // Add AI metadata
                if (isAI) {
                    const confidenceElement = message.querySelector('.confidence-text');
                    const architectureElement = message.querySelector('.architecture-badge');

                    if (confidenceElement) {
                        htmlContent += `<span>Confidence: ${confidenceElement.textContent}</span> `;
                    }
                    if (architectureElement) {
                        htmlContent += `<span>Architecture: ${architectureElement.textContent}</span>`;
                    }
                }

                htmlContent += `
            </div>
        </div>`;
            });

            htmlContent += `
    </div>
</body>
</html>`;

            downloadFile(htmlContent, `chat-export-${getTimestamp()}.html`, 'text/html');

        } catch (error) {
            console.error('HTML export error:', error);
            alert('Failed to export chat as HTML. Please try again.');
        }
    }

    // Export chat as JSON file
    function exportChatAsJSON() {
        try {
            const messages = document.querySelectorAll('#chatbot-messages .message');
            if (messages.length === 0) {
                alert('No chat messages to export.');
                return;
            }

            const chatData = {
                exportDate: new Date().toISOString(),
                exportTimestamp: Date.now(),
                messages: []
            };

            messages.forEach((message, index) => {
                const isUser = message.classList.contains('user-message');
                const isAI = message.classList.contains('ai-message');
                const isSystem = message.classList.contains('system-message');

                if (!isUser && !isAI && !isSystem) return;

                const messageTextElement = message.querySelector('.message-text');
                const timestampElement = message.querySelector('.timestamp');

                if (!messageTextElement) return;

                const messageText = messageTextElement.textContent || messageTextElement.innerText || '';
                const timestamp = timestampElement ? timestampElement.textContent : '';

                let messageType = 'unknown';
                if (isUser) messageType = 'user';
                else if (isAI) messageType = 'ai';
                else if (isSystem) messageType = 'system';

                const messageData = {
                    index: index,
                    type: messageType,
                    text: messageText.trim(),
                    timestamp: timestamp
                };

                // Add AI metadata
                if (isAI) {
                    const confidenceElement = message.querySelector('.confidence-text');
                    const architectureElement = message.querySelector('.architecture-badge');

                    if (confidenceElement) {
                        messageData.confidence = confidenceElement.textContent;
                    }
                    if (architectureElement) {
                        messageData.architecture = architectureElement.textContent;
                    }
                }

                chatData.messages.push(messageData);
            });

            const jsonContent = JSON.stringify(chatData, null, 2);
            downloadFile(jsonContent, `chat-export-${getTimestamp()}.json`, 'application/json');

        } catch (error) {
            console.error('JSON export error:', error);
            alert('Failed to export chat as JSON. Please try again.');
        }
    }

    // Helper function to download files
    function downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);

        // Show success message
        showSuccessMessage(`Chat exported successfully as ${filename}`);
    }

    // Helper function to get timestamp for filenames
    function getTimestamp() {
        return new Date().toISOString().slice(0, 19).replace(/:/g, '-');
    }

    // Helper function to show success messages
    function showSuccessMessage(message) {
        const successDiv = document.createElement('div');
        successDiv.textContent = message;
        successDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #28a745;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            z-index: 10000;
            font-size: 14px;
        `;
        document.body.appendChild(successDiv);

        setTimeout(() => {
            if (document.body.contains(successDiv)) {
                document.body.removeChild(successDiv);
            }
        }, 3000);
    }

    // Document Manager Functions
    function openDocumentManager() {
        const modal = document.getElementById('document-manager-modal');
        if (!modal) {
            console.error('Document manager modal not found');
            return;
        }
        modal.style.display = 'block';
        loadDocumentList();
    }

    function closeDocumentManager() {
        const modal = document.getElementById('document-manager-modal');
        if (modal) {
            modal.style.display = 'none';
        }
    }

    async function loadDocumentList() {
        const loadingDiv = document.getElementById('document-list-loading');
        const containerDiv = document.getElementById('document-list-container');
        const emptyDiv = document.getElementById('document-list-empty');
        const listDiv = document.getElementById('document-list');

        // Show loading state
        if (loadingDiv) loadingDiv.style.display = 'block';
        if (containerDiv) containerDiv.style.display = 'none';
        if (emptyDiv) emptyDiv.style.display = 'none';

        try {
            const response = await fetch('/api/documents_detailed');
            const data = await response.json();

            if (!data.success) {
                throw new Error(data.error || 'Failed to load documents');
            }

            // Hide loading
            if (loadingDiv) loadingDiv.style.display = 'none';

            if (data.documents && data.documents.length > 0) {
                // Show document list
                if (containerDiv) containerDiv.style.display = 'block';

                // Update stats
                const totalChunks = data.documents.reduce((sum, doc) => sum + (doc.chunk_count || 0), 0);
                document.getElementById('doc-count').textContent = `${data.documents.length} documents`;
                document.getElementById('chunk-count').textContent = `${totalChunks} chunks`;

                // Build document list HTML
                let html = '';
                data.documents.forEach(doc => {
                    const uploadDate = doc.created_at ? new Date(doc.created_at).toLocaleDateString() : 'Unknown';
                    const fileSize = doc.file_size_mb || 0;

                    html += `
                        <div class="document-item" style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 10px; background: white;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="flex: 1;">
                                    <h4 style="margin: 0 0 5px 0; color: #333;">
                                        <i class="fas fa-file-pdf" style="color: #dc3545;"></i>
                                        ${doc.display_name || doc.filename}
                                    </h4>
                                    <div style="color: #666; font-size: 0.9em;">
                                        <span style="margin-right: 15px;">
                                            <i class="fas fa-file"></i> ${doc.filename}
                                        </span>
                                        <span style="margin-right: 15px;">
                                            <i class="fas fa-cube"></i> ${doc.chunk_count} chunks
                                        </span>
                                        <span style="margin-right: 15px;">
                                            <i class="fas fa-calendar"></i> ${uploadDate}
                                        </span>
                                        ${fileSize > 0 ? `<span><i class="fas fa-weight"></i> ${fileSize.toFixed(2)} MB</span>` : ''}
                                    </div>
                                </div>
                                <div>
                                    <button onclick="deleteDocument(${doc.id}, '${doc.filename.replace(/'/g, "\\'")}')"
                                            class="btn-danger"
                                            style="padding: 8px 15px; border-radius: 5px; background: #dc3545; color: white; border: none; cursor: pointer;">
                                        <i class="fas fa-trash"></i> Delete
                                    </button>
                                </div>
                            </div>
                        </div>
                    `;
                });

                listDiv.innerHTML = html;
            } else {
                // Show empty state
                if (emptyDiv) emptyDiv.style.display = 'block';
            }

        } catch (error) {
            console.error('Error loading documents:', error);
            if (loadingDiv) loadingDiv.style.display = 'none';
            if (emptyDiv) {
                emptyDiv.style.display = 'block';
                emptyDiv.innerHTML = `
                    <i class="fas fa-exclamation-triangle fa-3x" style="color: #dc3545;"></i>
                    <p style="margin-top: 20px; color: #dc3545;">Error loading documents</p>
                    <p style="color: #666;">${error.message}</p>
                `;
            }
        }
    }

    async function deleteDocument(documentId, filename) {
        if (!confirm(`Are you sure you want to delete "${filename}"?\n\nThis action cannot be undone.`)) {
            return;
        }

        try {
            // Delete the document
            const response = await fetch('/api/delete_document', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    document_id: documentId,
                    filename: filename
                })
            });

            const data = await response.json();

            if (!data.success) {
                throw new Error(data.error || 'Failed to delete document');
            }

            // Show success message
            addMessage(`✅ Document "${filename}" deleted successfully`, 'system');

            // Trigger index rebuild in background
            rebuildIndexesInBackground();

            // Reload document list
            loadDocumentList();

            // Update document filter if needed
            if (window.updateDocumentFilter) {
                window.updateDocumentFilter();
            }

        } catch (error) {
            console.error('Error deleting document:', error);
            alert(`Failed to delete document: ${error.message}`);
        }
    }

    async function rebuildIndexesInBackground() {
        try {
            // Start index rebuild
            const response = await fetch('/api/rebuild_indexes', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const data = await response.json();

            if (data.success) {
                console.log('Indexes rebuilt successfully:', data.indexes);
                addMessage('🔄 Search indexes updated', 'system');
            } else {
                console.error('Index rebuild partially failed:', data);
            }
        } catch (error) {
            console.error('Error rebuilding indexes:', error);
            // Don't show error to user as this is a background operation
        }
    }

    function refreshDocumentList() {
        loadDocumentList();
    }

    // Make document manager functions globally accessible
    window.openDocumentManager = openDocumentManager;
    window.closeDocumentManager = closeDocumentManager;
    window.deleteDocument = deleteDocument;
    window.refreshDocumentList = refreshDocumentList;

    // Make export functions globally accessible
    window.exportChatAsPDF = exportChatAsPDF;
    window.exportChatAsHTML = exportChatAsHTML;
    window.exportChatAsJSON = exportChatAsJSON;
});

// Toggle reasoning section
function toggleReasoning(button) {
    const reasoningContent = button.parentElement.nextElementSibling;
    const icon = button.querySelector('i');

    if (reasoningContent.style.display === 'none') {
        reasoningContent.style.display = 'block';
        icon.className = 'fas fa-chevron-up';
    } else {
        reasoningContent.style.display = 'none';
        icon.className = 'fas fa-chevron-down';
    }
}

// Helper functions for dashboard integration
function askAI(question) {
    const chatbotInput = document.getElementById('chatbot-input');
    const chatbotWindow = document.getElementById('chatbot-window');

    if (chatbotInput && chatbotWindow) {
        // Open chatbot if closed
        if (chatbotWindow.style.display === 'none') {
            document.getElementById('chatbot-toggle').click();
        }

        // Set question and send
        chatbotInput.value = question;
        chatbotInput.focus();
        document.getElementById('chatbot-send').click();
    }
}

function getChatbotContextInfo() {
    // Detect current page for context-aware responses
    const path = window.location.pathname;
    const pageName = path.split('/').pop() || 'dashboard';

    return {
        page: pageName,
        context: getPageContext(pageName)
    };
}

function getPageContext(pageName) {
    const contexts = {
        'fault-overview': 'fault monitoring and analysis',
        'performance-overview': 'performance metrics and KPIs',
        'network-topology': 'network topology and connections',
        'security-overview': 'security metrics and alerts',
        'alarm-list': 'alarm management',
        'alert-list': 'alert management'
    };

    return contexts[pageName] || 'general dashboard';
}

    // Mode button event listeners
    const modeButtons = document.querySelectorAll('.mode-btn');

    modeButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const mode = this.getAttribute('data-mode');

            if (mode && typeof window.setMode === 'function') {
                window.setMode(mode);
            }
        });
    });

    // Set initial placeholder
    const input = document.getElementById('chatbot-input');
    if (input && window.modeConfig && window.currentMode) {
        input.placeholder = window.modeConfig[window.currentMode].placeholder;
    }

    // Initialize document filter with existing documents
    document.addEventListener('DOMContentLoaded', function() {
        updateDocumentFilter();
    });

    // Also update when SOP mode is activated
    document.addEventListener('DOMContentLoaded', function() {
        const sopModeBtn = document.getElementById('sop-mode-btn');
        if (sopModeBtn) {
            sopModeBtn.addEventListener('click', function() {
                setTimeout(updateDocumentFilter, 100); // Small delay to ensure mode is set
            });
        }
    });