// Minimal Voice-to-Text for Chatbot
// YAGNI: Only essential features - mic button toggles recording, text goes to input

class VoiceRecognition {
    constructor() {
        this.recognition = null;
        this.isListening = false;
        this.transcript = '';

        this.init();
    }

    init() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

        if (!SpeechRecognition) {
            this.hideButton();
            return;
        }

        this.recognition = new SpeechRecognition();
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = 'en-US';

        this.recognition.onresult = (event) => {
            this.transcript = '';
            for (let i = 0; i < event.results.length; i++) {
                this.transcript += event.results[i][0].transcript;
            }
            this.updateInput();
        };

        this.recognition.onerror = (event) => {
            if (event.error === 'not-allowed') {
                alert('Microphone access denied. Please allow microphone access.');
            }
            this.stop();
        };

        this.recognition.onend = () => {
            if (this.isListening) {
                this.recognition.start();
            }
        };
    }

    start() {
        if (!this.recognition) return;

        this.transcript = '';
        this.recognition.start();
        this.isListening = true;
        this.updateButton();
    }

    stop() {
        if (!this.recognition) return;

        this.isListening = false;
        this.recognition.stop();
        this.updateButton();
    }

    updateInput() {
        const input = document.getElementById('chatbot-input');
        if (input) input.value = this.transcript;
    }

    updateButton() {
        const btn = document.getElementById('voice-btn');
        const icon = document.getElementById('voice-icon');

        if (!btn || !icon) return;

        if (this.isListening) {
            btn.classList.add('listening');
            icon.className = 'fas fa-stop-circle';
        } else {
            btn.classList.remove('listening');
            icon.className = 'fas fa-microphone';
        }
    }

    hideButton() {
        const btn = document.getElementById('voice-btn');
        if (btn) btn.style.display = 'none';
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    const voiceInput = new VoiceRecognition();

    const voiceBtn = document.getElementById('voice-btn');
    if (voiceBtn) {
        voiceBtn.addEventListener('click', () => {
            if (voiceInput.isListening) {
                voiceInput.stop();
            } else {
                voiceInput.start();
            }
        });
    }

    window.voiceInput = voiceInput;
});
