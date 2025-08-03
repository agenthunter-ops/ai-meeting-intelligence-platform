# AI Meeting Intelligence Platform - Part 4: Frontend Implementation & User Experience

## Table of Contents
1. [Frontend Architecture Overview](#frontend-architecture-overview)
2. [Static HTML Implementation Strategy](#static-html-implementation-strategy)
3. [User Interface Design Patterns](#user-interface-design-patterns)
4. [Real-time Communication](#real-time-communication)
5. [State Management](#state-management)
6. [Progressive Enhancement](#progressive-enhancement)
7. [Accessibility and Responsiveness](#accessibility-and-responsiveness)
8. [Performance Optimization](#performance-optimization)

---

## Frontend Architecture Overview

### Modern Static Frontend Architecture

Our frontend represents a strategic decision to prioritize reliability, performance, and maintainability over framework complexity. Instead of a traditional Single Page Application (SPA), we've implemented a **Progressive Static Application** that delivers enterprise-grade functionality while maintaining simplicity.

```html
<!-- frontend/index-simple.html - Core Application Structure -->

<!DOCTYPE html>
<html lang="en" class="h-full">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Meeting Intelligence Platform</title>
    
    <!-- Tailwind CSS for utility-first styling -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Custom Tailwind Configuration -->
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        'brand-blue': '#3b82f6',
                        'brand-indigo': '#6366f1',
                        'success-green': '#10b981',
                        'warning-amber': '#f59e0b',
                        'error-red': '#ef4444'
                    },
                    fontFamily: {
                        'system': ['-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif']
                    },
                    animation: {
                        'fade-in': 'fadeIn 0.5s ease-in-out',
                        'slide-up': 'slideUp 0.3s ease-out',
                        'pulse-slow': 'pulse 2s infinite'
                    }
                }
            }
        }
    </script>
    
    <!-- Performance optimizations -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="dns-prefetch" href="/api">
    
    <!-- Meta tags for SEO and social sharing -->
    <meta name="description" content="AI-powered meeting intelligence platform for transcription, analysis, and insights">
    <meta name="keywords" content="meeting, AI, transcription, analysis, collaboration">
    <meta property="og:title" content="AI Meeting Intelligence Platform">
    <meta property="og:description" content="Transform your meetings into actionable insights">
    <meta property="og:type" content="website">
    
    <!-- Favicon and app icons -->
    <link rel="icon" type="image/x-icon" href="/favicon.ico">
    <meta name="theme-color" content="#3b82f6">
    
    <!-- Styles for enhanced UI components -->
    <style>
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes slideUp {
            from { 
                opacity: 0; 
                transform: translateY(20px); 
            }
            to { 
                opacity: 1; 
                transform: translateY(0); 
            }
        }
        
        /* Custom component styles */
        .card {
            @apply bg-white rounded-xl shadow-lg border border-gray-100 hover:shadow-xl transition-all duration-300;
        }
        
        .btn-primary {
            @apply bg-gradient-to-r from-brand-blue to-brand-indigo text-white font-semibold py-3 px-6 rounded-lg 
                   hover:from-blue-600 hover:to-indigo-600 transform hover:scale-105 transition-all duration-200 
                   focus:outline-none focus:ring-4 focus:ring-blue-300 disabled:opacity-50 disabled:cursor-not-allowed;
        }
        
        .btn-secondary {
            @apply bg-gray-100 text-gray-700 font-semibold py-3 px-6 rounded-lg 
                   hover:bg-gray-200 transition-all duration-200 focus:outline-none focus:ring-4 focus:ring-gray-300;
        }
        
        .input-field {
            @apply w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-4 focus:ring-blue-100 
                   focus:border-brand-blue transition-all duration-200 placeholder-gray-400;
        }
        
        .progress-bar {
            @apply bg-gray-200 rounded-full overflow-hidden;
        }
        
        .progress-fill {
            @apply bg-gradient-to-r from-brand-blue to-brand-indigo h-full transition-all duration-500 ease-out;
        }
        
        .status-badge {
            @apply inline-flex items-center px-3 py-1 rounded-full text-sm font-medium;
        }
        
        .status-processing {
            @apply bg-amber-100 text-amber-800;
        }
        
        .status-completed {
            @apply bg-green-100 text-green-800;
        }
        
        .status-error {
            @apply bg-red-100 text-red-800;
        }
        
        /* Loading animations */
        .spinner {
            @apply inline-block w-6 h-6 border-4 border-current border-t-transparent rounded-full animate-spin;
        }
        
        /* Responsive grid layouts */
        .grid-auto-fit {
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        }
        
        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            .card {
                @apply bg-gray-800 border-gray-700 text-white;
            }
        }
        
        /* Custom scrollbar */
        .custom-scrollbar::-webkit-scrollbar {
            width: 8px;
        }
        
        .custom-scrollbar::-webkit-scrollbar-track {
            @apply bg-gray-100 rounded-lg;
        }
        
        .custom-scrollbar::-webkit-scrollbar-thumb {
            @apply bg-gray-400 rounded-lg hover:bg-gray-500;
        }
        
        /* Accessibility improvements */
        .sr-only {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        }
        
        /* Focus styles for keyboard navigation */
        .focus-visible:focus {
            @apply outline-none ring-4 ring-brand-blue ring-opacity-50;
        }
    </style>
</head>

<body class="bg-gray-50 font-system min-h-screen flex flex-col">
    <!-- Accessibility skip link -->
    <a href="#main-content" class="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 bg-brand-blue text-white p-2 rounded">
        Skip to main content
    </a>
    
    <!-- Navigation Header -->
    <nav class="bg-white shadow-lg border-b border-gray-200" role="navigation" aria-label="Main navigation">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between items-center h-16">
                <!-- Logo and brand -->
                <div class="flex items-center space-x-4">
                    <div class="flex-shrink-0">
                        <div class="w-10 h-10 bg-gradient-to-br from-brand-blue to-brand-indigo rounded-lg flex items-center justify-center">
                            <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                      d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/>
                            </svg>
                        </div>
                    </div>
                    <div>
                        <h1 class="text-xl font-bold text-gray-900">AI Meeting Intelligence</h1>
                        <p class="text-sm text-gray-500">Transform meetings into insights</p>
                    </div>
                </div>
                
                <!-- Navigation links -->
                <div class="hidden md:flex items-center space-x-8">
                    <a href="#dashboard" 
                       class="nav-link text-gray-700 hover:text-brand-blue px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200"
                       aria-current="page">
                        Dashboard
                    </a>
                    <a href="#upload" 
                       class="nav-link text-gray-700 hover:text-brand-blue px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200">
                        Upload
                    </a>
                    <a href="#search" 
                       class="nav-link text-gray-700 hover:text-brand-blue px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200">
                        Search
                    </a>
                    <a href="#analytics" 
                       class="nav-link text-gray-700 hover:text-brand-blue px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200">
                        Analytics
                    </a>
                </div>
                
                <!-- Mobile menu button -->
                <div class="md:hidden">
                    <button id="mobile-menu-button" 
                            class="text-gray-700 hover:text-brand-blue focus:outline-none focus:ring-2 focus:ring-brand-blue p-2 rounded-md"
                            aria-expanded="false" aria-controls="mobile-menu">
                        <span class="sr-only">Open main menu</span>
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/>
                        </svg>
                    </button>
                </div>
            </div>
            
            <!-- Mobile menu -->
            <div id="mobile-menu" class="md:hidden hidden">
                <div class="px-2 pt-2 pb-3 space-y-1 border-t border-gray-200">
                    <a href="#dashboard" class="mobile-nav-link block px-3 py-2 text-gray-700 hover:text-brand-blue rounded-md text-base font-medium">Dashboard</a>
                    <a href="#upload" class="mobile-nav-link block px-3 py-2 text-gray-700 hover:text-brand-blue rounded-md text-base font-medium">Upload</a>
                    <a href="#search" class="mobile-nav-link block px-3 py-2 text-gray-700 hover:text-brand-blue rounded-md text-base font-medium">Search</a>
                    <a href="#analytics" class="mobile-nav-link block px-3 py-2 text-gray-700 hover:text-brand-blue rounded-md text-base font-medium">Analytics</a>
                </div>
            </div>
        </div>
    </nav>
    
    <!-- Main Content Area -->
    <main id="main-content" class="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8" role="main">
        <!-- Global Status Bar -->
        <div id="global-status" class="hidden mb-6">
            <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <div class="flex items-center">
                    <div class="spinner text-brand-blue mr-3"></div>
                    <div>
                        <p class="text-sm font-medium text-blue-800" id="status-message">Processing...</p>
                        <div class="mt-2 w-full bg-blue-200 rounded-full h-2">
                            <div id="global-progress" class="progress-fill h-2 rounded-full" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
```

### Component-Based Architecture Without Frameworks

Despite being a static application, we implement component-like patterns using vanilla JavaScript:

```javascript
// Application Architecture - Modular Component System

class ComponentManager {
    constructor() {
        this.components = new Map();
        this.eventBus = new EventTarget();
        this.state = new Map();
    }
    
    registerComponent(name, component) {
        this.components.set(name, component);
        component.eventBus = this.eventBus;
        component.setState = this.setState.bind(this);
        component.getState = this.getState.bind(this);
    }
    
    setState(key, value) {
        const oldValue = this.state.get(key);
        this.state.set(key, value);
        
        // Emit state change event
        this.eventBus.dispatchEvent(new CustomEvent('stateChange', {
            detail: { key, value, oldValue }
        }));
    }
    
    getState(key) {
        return this.state.get(key);
    }
    
    emit(eventName, data) {
        this.eventBus.dispatchEvent(new CustomEvent(eventName, { detail: data }));
    }
    
    on(eventName, handler) {
        this.eventBus.addEventListener(eventName, handler);
    }
    
    off(eventName, handler) {
        this.eventBus.removeEventListener(eventName, handler);
    }
}

// Base Component Class
class BaseComponent {
    constructor(element) {
        this.element = element;
        this.eventBus = null;
        this.setState = null;
        this.getState = null;
        this.isVisible = false;
        
        this.init();
    }
    
    init() {
        // Override in subclasses
    }
    
    show() {
        this.element.classList.remove('hidden');
        this.element.classList.add('animate-fade-in');
        this.isVisible = true;
        this.onShow();
    }
    
    hide() {
        this.element.classList.add('hidden');
        this.element.classList.remove('animate-fade-in');
        this.isVisible = false;
        this.onHide();
    }
    
    onShow() {
        // Override in subclasses
    }
    
    onHide() {
        // Override in subclasses
    }
    
    updateUI() {
        // Override in subclasses
    }
    
    destroy() {
        // Cleanup when component is removed
        if (this.element) {
            this.element.remove();
        }
    }
}

// Dashboard Component
class DashboardComponent extends BaseComponent {
    init() {
        this.bindEvents();
        this.loadRecentMeetings();
        this.updateStats();
    }
    
    bindEvents() {
        // Listen for upload completion
        this.eventBus?.addEventListener('uploadComplete', (event) => {
            this.loadRecentMeetings();
            this.updateStats();
        });
        
        // Listen for processing updates
        this.eventBus?.addEventListener('processingUpdate', (event) => {
            this.updateProcessingStatus(event.detail);
        });
    }
    
    async loadRecentMeetings() {
        try {
            const response = await fetch('/api/meetings?limit=5&sort=recent');
            if (response.ok) {
                const meetings = await response.json();
                this.renderRecentMeetings(meetings);
            }
        } catch (error) {
            console.error('Failed to load recent meetings:', error);
            this.showError('Failed to load recent meetings');
        }
    }
    
    renderRecentMeetings(meetings) {
        const container = this.element.querySelector('#recent-meetings');
        if (!container) return;
        
        if (meetings.length === 0) {
            container.innerHTML = `
                <div class="text-center py-8">
                    <svg class="w-12 h-12 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                              d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/>
                    </svg>
                    <p class="text-gray-500">No meetings yet. Upload your first meeting to get started!</p>
                    <button onclick="app.showSection('upload')" class="btn-primary mt-4">
                        Upload Meeting
                    </button>
                </div>
            `;
            return;
        }
        
        const meetingsHTML = meetings.map(meeting => `
            <div class="card p-4 hover:bg-gray-50 cursor-pointer transition-colors duration-200"
                 onclick="app.viewMeeting('${meeting.id}')">
                <div class="flex items-center justify-between">
                    <div class="flex-1">
                        <h4 class="font-semibold text-gray-900 truncate">${this.escapeHtml(meeting.title)}</h4>
                        <p class="text-sm text-gray-500 mt-1">
                            ${this.formatDate(meeting.upload_time)} • ${this.formatDuration(meeting.duration)}
                        </p>
                        <div class="flex items-center mt-2 space-x-4">
                            <span class="status-badge ${this.getStatusClass(meeting.status)}">${meeting.status}</span>
                            ${meeting.speaker_count ? `<span class="text-xs text-gray-500">${meeting.speaker_count} speakers</span>` : ''}
                        </div>
                    </div>
                    <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/>
                    </svg>
                </div>
            </div>
        `).join('');
        
        container.innerHTML = meetingsHTML;
    }
    
    async updateStats() {
        try {
            const response = await fetch('/api/stats');
            if (response.ok) {
                const stats = await response.json();
                this.renderStats(stats);
            }
        } catch (error) {
            console.error('Failed to load stats:', error);
        }
    }
    
    renderStats(stats) {
        const statsContainer = this.element.querySelector('#dashboard-stats');
        if (!statsContainer) return;
        
        const statsHTML = `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div class="card p-6 text-center">
                    <div class="text-3xl font-bold text-brand-blue mb-2">${stats.total_meetings || 0}</div>
                    <div class="text-gray-600">Total Meetings</div>
                    <div class="text-sm text-gray-500 mt-1">
                        ${stats.meetings_this_month || 0} this month
                    </div>
                </div>
                <div class="card p-6 text-center">
                    <div class="text-3xl font-bold text-success-green mb-2">${this.formatTime(stats.total_duration || 0)}</div>
                    <div class="text-gray-600">Total Duration</div>
                    <div class="text-sm text-gray-500 mt-1">
                        ${this.formatTime(stats.duration_this_month || 0)} this month
                    </div>
                </div>
                <div class="card p-6 text-center">
                    <div class="text-3xl font-bold text-purple-600 mb-2">${stats.processing_queue || 0}</div>
                    <div class="text-gray-600">Processing Queue</div>
                    <div class="text-sm text-gray-500 mt-1">
                        ${stats.completed_today || 0} completed today
                    </div>
                </div>
            </div>
        `;
        
        statsContainer.innerHTML = statsHTML;
    }
    
    updateProcessingStatus(status) {
        const globalStatus = document.getElementById('global-status');
        const statusMessage = document.getElementById('status-message');
        const globalProgress = document.getElementById('global-progress');
        
        if (status.active_tasks > 0) {
            globalStatus.classList.remove('hidden');
            statusMessage.textContent = `Processing ${status.active_tasks} meeting(s)...`;
            globalProgress.style.width = `${status.average_progress}%`;
        } else {
            globalStatus.classList.add('hidden');
        }
    }
    
    // Utility methods
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    formatDate(dateString) {
        if (!dateString) return 'Unknown date';
        const date = new Date(dateString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    }
    
    formatDuration(seconds) {
        if (!seconds) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    
    formatTime(seconds) {
        if (!seconds) return '0h 0m';
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${minutes}m`;
    }
    
    getStatusClass(status) {
        switch (status) {
            case 'completed': return 'status-completed';
            case 'processing': return 'status-processing';
            case 'failed': return 'status-error';
            default: return 'bg-gray-100 text-gray-800';
        }
    }
    
    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'bg-red-50 border border-red-200 rounded-lg p-4 mb-4';
        errorDiv.innerHTML = `
            <div class="flex items-center">
                <svg class="w-5 h-5 text-red-400 mr-3" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>
                </svg>
                <p class="text-red-800">${message}</p>
            </div>
        `;
        
        this.element.insertBefore(errorDiv, this.element.firstChild);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }
}

// Upload Component
class UploadComponent extends BaseComponent {
    init() {
        this.setupFileUpload();
        this.setupFormValidation();
        this.bindEvents();
        this.currentTaskId = null;
        this.pollInterval = null;
    }
    
    setupFileUpload() {
        const fileInput = this.element.querySelector('#fileInput');
        const dropZone = this.element.querySelector('#drop-zone');
        const uploadForm = this.element.querySelector('#upload-form');
        
        // File input change handler
        fileInput?.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.handleFileSelection(file);
            }
        });
        
        // Drag and drop functionality
        if (dropZone) {
            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.classList.add('border-brand-blue', 'bg-blue-50');
            });
            
            dropZone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                dropZone.classList.remove('border-brand-blue', 'bg-blue-50');
            });
            
            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('border-brand-blue', 'bg-blue-50');
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    this.handleFileSelection(files[0]);
                }
            });
        }
        
        // Form submission
        uploadForm?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.uploadFile();
        });
    }
    
    handleFileSelection(file) {
        // Validate file
        const validation = this.validateFile(file);
        if (!validation.valid) {
            this.showError(validation.error);
            return;
        }
        
        // Update UI to show selected file
        this.updateFilePreview(file);
        
        // Enable upload button
        const uploadButton = this.element.querySelector('#upload-button');
        if (uploadButton) {
            uploadButton.disabled = false;
            uploadButton.textContent = 'Upload & Process';
        }
        
        this.selectedFile = file;
    }
    
    validateFile(file) {
        const allowedTypes = [
            'audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/mp4',
            'video/mp4', 'video/quicktime', 'video/x-msvideo'
        ];
        
        const allowedExtensions = ['.mp3', '.wav', '.m4a', '.mp4', '.mov', '.avi'];
        
        // Check file size (500MB limit)
        const maxSize = 500 * 1024 * 1024;
        if (file.size > maxSize) {
            return {
                valid: false,
                error: 'File size exceeds 500MB limit'
            };
        }
        
        // Check file type
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        if (!allowedExtensions.includes(fileExtension)) {
            return {
                valid: false,
                error: `Unsupported file type. Allowed: ${allowedExtensions.join(', ')}`
            };
        }
        
        return { valid: true };
    }
    
    updateFilePreview(file) {
        const preview = this.element.querySelector('#file-preview');
        if (!preview) return;
        
        const fileSize = this.formatFileSize(file.size);
        const fileType = file.type || 'Unknown';
        
        preview.innerHTML = `
            <div class="card p-4 border-brand-blue bg-blue-50">
                <div class="flex items-center space-x-3">
                    <div class="flex-shrink-0">
                        <svg class="w-8 h-8 text-brand-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                  d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"/>
                        </svg>
                    </div>
                    <div class="flex-1 min-w-0">
                        <p class="font-medium text-gray-900 truncate">${file.name}</p>
                        <p class="text-sm text-gray-500">${fileSize} • ${fileType}</p>
                    </div>
                    <button type="button" onclick="app.clearFileSelection()" 
                            class="text-gray-400 hover:text-gray-600">
                        <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"/>
                        </svg>
                    </button>
                </div>
            </div>
        `;
        
        preview.classList.remove('hidden');
    }
    
    async uploadFile() {
        if (!this.selectedFile) {
            this.showError('Please select a file first');
            return;
        }
        
        const title = this.element.querySelector('#meeting-title')?.value || '';
        const meetingType = this.element.querySelector('#meeting-type')?.value || 'general';
        
        // Show upload progress
        this.showUploadProgress();
        
        try {
            const formData = new FormData();
            formData.append('file', this.selectedFile);
            formData.append('title', title);
            formData.append('meeting_type', meetingType);
            
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                this.currentTaskId = result.task_id;
                this.showProcessingStatus(result);
                this.startStatusPolling();
                
                // Emit upload event
                this.eventBus?.dispatchEvent(new CustomEvent('uploadStarted', {
                    detail: { taskId: result.task_id, meetingId: result.meeting_id }
                }));
                
            } else {
                const error = await response.json();
                throw new Error(error.detail || 'Upload failed');
            }
            
        } catch (error) {
            console.error('Upload failed:', error);
            this.showError(`Upload failed: ${error.message}`);
            this.hideUploadProgress();
        }
    }
    
    showUploadProgress() {
        const progressSection = this.element.querySelector('#upload-progress');
        const uploadButton = this.element.querySelector('#upload-button');
        
        if (progressSection) {
            progressSection.classList.remove('hidden');
        }
        
        if (uploadButton) {
            uploadButton.disabled = true;
            uploadButton.innerHTML = `
                <div class="spinner mr-2"></div>
                Uploading...
            `;
        }
    }
    
    hideUploadProgress() {
        const progressSection = this.element.querySelector('#upload-progress');
        const uploadButton = this.element.querySelector('#upload-button');
        
        if (progressSection) {
            progressSection.classList.add('hidden');
        }
        
        if (uploadButton) {
            uploadButton.disabled = false;
            uploadButton.innerHTML = 'Upload & Process';
        }
    }
    
    showProcessingStatus(result) {
        const statusSection = this.element.querySelector('#processing-status');
        if (!statusSection) return;
        
        statusSection.innerHTML = `
            <div class="card p-6 border-brand-blue">
                <div class="text-center">
                    <div class="w-16 h-16 bg-brand-blue rounded-full flex items-center justify-center mx-auto mb-4">
                        <svg class="w-8 h-8 text-white animate-spin" fill="none" viewBox="0 0 24 24">
                            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                    </div>
                    <h3 class="text-lg font-semibold text-gray-900 mb-2">Processing Meeting</h3>
                    <p class="text-gray-600 mb-4">Your meeting is being analyzed. This may take a few minutes.</p>
                    
                    <!-- Progress bar -->
                    <div class="progress-bar h-3 mb-4">
                        <div id="processing-progress" class="progress-fill" style="width: 0%"></div>
                    </div>
                    
                    <div class="text-sm text-gray-500 space-y-1">
                        <p id="processing-stage">Starting processing...</p>
                        <p>Task ID: <code class="bg-gray-100 px-2 py-1 rounded text-xs">${result.task_id}</code></p>
                    </div>
                </div>
            </div>
        `;
        
        statusSection.classList.remove('hidden');
    }
    
    startStatusPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
        }
        
        this.pollInterval = setInterval(() => {
            this.pollTaskStatus();
        }, 2000); // Poll every 2 seconds
    }
    
    async pollTaskStatus() {
        if (!this.currentTaskId) return;
        
        try {
            const response = await fetch(`/api/status/${this.currentTaskId}`);
            if (response.ok) {
                const status = await response.json();
                this.updateProcessingStatus(status);
                
                if (status.status === 'completed' || status.status === 'failed') {
                    this.stopStatusPolling();
                    this.handleProcessingComplete(status);
                }
            }
        } catch (error) {
            console.error('Failed to poll status:', error);
        }
    }
    
    updateProcessingStatus(status) {
        const progressBar = this.element.querySelector('#processing-progress');
        const stageText = this.element.querySelector('#processing-stage');
        
        if (progressBar) {
            progressBar.style.width = `${status.progress}%`;
        }
        
        if (stageText) {
            const stageMessages = {
                'pending': 'Waiting to start...',
                'running': this.getStageMessage(status.progress),
                'completed': 'Processing completed!',
                'failed': 'Processing failed'
            };
            
            stageText.textContent = stageMessages[status.status] || 'Processing...';
        }
    }
    
    getStageMessage(progress) {
        if (progress < 20) return 'Analyzing audio quality...';
        if (progress < 40) return 'Performing speaker diarization...';
        if (progress < 70) return 'Transcribing speech...';
        if (progress < 90) return 'Analyzing sentiment and emotions...';
        return 'Generating summary and insights...';
    }
    
    handleProcessingComplete(status) {
        const statusSection = this.element.querySelector('#processing-status');
        if (!statusSection) return;
        
        if (status.status === 'completed') {
            statusSection.innerHTML = `
                <div class="card p-6 border-green-300 bg-green-50">
                    <div class="text-center">
                        <div class="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-4">
                            <svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
                            </svg>
                        </div>
                        <h3 class="text-lg font-semibold text-green-900 mb-2">Processing Complete!</h3>
                        <p class="text-green-700 mb-4">Your meeting has been successfully analyzed.</p>
                        
                        <div class="space-y-3">
                            <button onclick="app.viewMeeting('${status.meeting_id}')" 
                                    class="btn-primary w-full">
                                View Meeting Analysis
                            </button>
                            <button onclick="app.uploadAnother()" 
                                    class="btn-secondary w-full">
                                Upload Another Meeting
                            </button>
                        </div>
                    </div>
                </div>
            `;
            
            // Emit completion event
            this.eventBus?.dispatchEvent(new CustomEvent('uploadComplete', {
                detail: { taskId: this.currentTaskId, meetingId: status.meeting_id }
            }));
            
        } else {
            // Show error
            statusSection.innerHTML = `
                <div class="card p-6 border-red-300 bg-red-50">
                    <div class="text-center">
                        <div class="w-16 h-16 bg-red-500 rounded-full flex items-center justify-center mx-auto mb-4">
                            <svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                            </svg>
                        </div>
                        <h3 class="text-lg font-semibold text-red-900 mb-2">Processing Failed</h3>
                        <p class="text-red-700 mb-4">${status.error_message || 'An error occurred during processing.'}</p>
                        
                        <button onclick="app.uploadAnother()" class="btn-primary">
                            Try Again
                        </button>
                    </div>
                </div>
            `;
        }
    }
    
    stopStatusPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }
    
    setupFormValidation() {
        const form = this.element.querySelector('#upload-form');
        const inputs = form?.querySelectorAll('input, select, textarea');
        
        inputs?.forEach(input => {
            input.addEventListener('blur', () => {
                this.validateField(input);
            });
            
            input.addEventListener('input', () => {
                this.clearFieldError(input);
            });
        });
    }
    
    validateField(field) {
        const value = field.value.trim();
        let isValid = true;
        let errorMessage = '';
        
        // Title validation
        if (field.id === 'meeting-title') {
            if (value.length > 255) {
                isValid = false;
                errorMessage = 'Title must be less than 255 characters';
            }
        }
        
        if (!isValid) {
            this.showFieldError(field, errorMessage);
        } else {
            this.clearFieldError(field);
        }
        
        return isValid;
    }
    
    showFieldError(field, message) {
        this.clearFieldError(field);
        
        field.classList.add('border-red-500', 'focus:ring-red-300');
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'field-error text-red-600 text-sm mt-1';
        errorDiv.textContent = message;
        
        field.parentNode.appendChild(errorDiv);
    }
    
    clearFieldError(field) {
        field.classList.remove('border-red-500', 'focus:ring-red-300');
        
        const existingError = field.parentNode.querySelector('.field-error');
        if (existingError) {
            existingError.remove();
        }
    }
    
    bindEvents() {
        // Listen for state changes
        this.eventBus?.addEventListener('stateChange', (event) => {
            if (event.detail.key === 'currentSection' && event.detail.value === 'upload') {
                this.onShow();
            }
        });
    }
    
    onShow() {
        // Reset form when shown
        this.resetForm();
    }
    
    resetForm() {
        const form = this.element.querySelector('#upload-form');
        form?.reset();
        
        const preview = this.element.querySelector('#file-preview');
        if (preview) {
            preview.classList.add('hidden');
        }
        
        const progressSection = this.element.querySelector('#upload-progress');
        if (progressSection) {
            progressSection.classList.add('hidden');
        }
        
        const statusSection = this.element.querySelector('#processing-status');
        if (statusSection) {
            statusSection.classList.add('hidden');
        }
        
        const uploadButton = this.element.querySelector('#upload-button');
        if (uploadButton) {
            uploadButton.disabled = true;
            uploadButton.textContent = 'Select File First';
        }
        
        this.selectedFile = null;
        this.currentTaskId = null;
        this.stopStatusPolling();
    }
    
    // Utility methods
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    showError(message) {
        // Similar to DashboardComponent.showError
        const errorDiv = document.createElement('div');
        errorDiv.className = 'bg-red-50 border border-red-200 rounded-lg p-4 mb-4';
        errorDiv.innerHTML = `
            <div class="flex items-center">
                <svg class="w-5 h-5 text-red-400 mr-3" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>
                </svg>
                <p class="text-red-800">${message}</p>
            </div>
        `;
        
        this.element.insertBefore(errorDiv, this.element.firstChild);
        
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }
    
    destroy() {
        this.stopStatusPolling();
        super.destroy();
    }
}
```

This completes the first major section of Part 4. The frontend architecture demonstrates a sophisticated approach to building modern web applications without complex frameworks, utilizing:

1. **Component-based architecture** using vanilla JavaScript classes
2. **Event-driven communication** between components
3. **Progressive enhancement** with accessibility features
4. **Real-time updates** through polling mechanisms
5. **Comprehensive error handling** and user feedback
6. **Modern CSS** with Tailwind utilities and custom animations

The implementation shows how to create enterprise-grade user experiences while maintaining simplicity and performance. Each component is designed to be modular, testable, and maintainable.

Would you like me to continue with the remaining sections of Part 4, covering Search Components, Real-time Communication, State Management, and Performance Optimization?