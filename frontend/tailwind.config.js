/**
 * AI Meeting Intelligence Platform - Tailwind CSS Configuration
 * ============================================================
 * 
 * This configuration file customizes Tailwind CSS for the Angular frontend.
 * It includes custom themes, component styles, and optimizations specifically
 * designed for a meeting intelligence dashboard interface.
 * 
 * Features:
 * - Custom color palette for meeting intelligence UI
 * - Dark mode support with system preference detection
 * - Custom animations for real-time updates
 * - Responsive design breakpoints
 * - Custom component utilities
 * - Optimized production build with purging
 */

/** @type {import('tailwindcss').Config} */
module.exports = {
    // Content sources for Tailwind to scan for classes
    content: [
      // Angular component templates
      "./src/**/*.{html,ts}",
      
      // Angular component TypeScript files (for dynamic classes)
      "./src/**/*.component.ts",
      
      // Angular services and modules (in case of dynamic class generation)
      "./src/**/*.service.ts",
      "./src/**/*.module.ts",
      
      // Index.html and any static HTML files
      "./src/index.html",
      
      // Include any additional template files
      "./src/**/*.template.html",
      
      // Include any JavaScript files that might contain classes
      "./src/**/*.js"
    ],
  
    // Dark mode configuration
    darkMode: 'class', // Enable class-based dark mode (can be 'media' for system preference)
  
    theme: {
      extend: {
        // Custom color palette for meeting intelligence theme
        colors: {
          // Primary brand colors
          primary: {
            50: '#eff6ff',   // Very light blue
            100: '#dbeafe',  // Light blue
            200: '#bfdbfe',  // Lighter blue
            300: '#93c5fd',  // Light blue
            400: '#60a5fa',  // Medium blue
            500: '#3b82f6',  // Primary blue (default)
            600: '#2563eb',  // Darker blue
            700: '#1d4ed8',  // Dark blue
            800: '#1e40af',  // Very dark blue
            900: '#1e3a8a',  // Darkest blue
            950: '#172554'   // Ultra dark blue
          },
  
          // Secondary accent colors
          secondary: {
            50: '#f8fafc',   // Very light gray
            100: '#f1f5f9',  // Light gray
            200: '#e2e8f0',  // Lighter gray
            300: '#cbd5e1',  // Light gray
            400: '#94a3b8',  // Medium gray
            500: '#64748b',  // Default gray
            600: '#475569',  // Darker gray
            700: '#334155',  // Dark gray
            800: '#1e293b',  // Very dark gray
            900: '#0f172a',  // Darkest gray
            950: '#020617'   // Ultra dark gray
          },
  
          // Success colors (for completed action items, positive sentiment)
          success: {
            50: '#f0fdf4',   // Very light green
            100: '#dcfce7',  // Light green
            200: '#bbf7d0',  // Lighter green
            300: '#86efac',  // Light green
            400: '#4ade80',  // Medium green
            500: '#22c55e',  // Default green
            600: '#16a34a',  // Darker green
            700: '#15803d',  // Dark green
            800: '#166534',  // Very dark green
            900: '#14532d',  // Darkest green
            950: '#052e16'   // Ultra dark green
          },
  
          // Warning colors (for pending items, neutral sentiment)
          warning: {
            50: '#fffbeb',   // Very light yellow
            100: '#fef3c7',  // Light yellow
            200: '#fde68a',  // Lighter yellow
            300: '#fcd34d',  // Light yellow
            400: '#fbbf24',  // Medium yellow
            500: '#f59e0b',  // Default yellow
            600: '#d97706',  // Darker yellow
            700: '#b45309',  // Dark yellow
            800: '#92400e',  // Very dark yellow
            900: '#78350f',  // Darkest yellow
            950: '#451a03'   // Ultra dark yellow
          },
  
          // Error colors (for failed tasks, negative sentiment)
          error: {
            50: '#fef2f2',   // Very light red
            100: '#fee2e2',  // Light red
            200: '#fecaca',  // Lighter red
            300: '#fca5a5',  // Light red
            400: '#f87171',  // Medium red
            500: '#ef4444',  // Default red
            600: '#dc2626',  // Darker red
            700: '#b91c1c',  // Dark red
            800: '#991b1b',  // Very dark red
            900: '#7f1d1d',  // Darkest red
            950: '#450a0a'   // Ultra dark red
          },
  
          // Info colors (for transcription status, general information)
          info: {
            50: '#f0f9ff',   // Very light cyan
            100: '#e0f2fe',  // Light cyan
            200: '#bae6fd',  // Lighter cyan
            300: '#7dd3fc',  // Light cyan
            400: '#38bdf8',  // Medium cyan
            500: '#0ea5e9',  // Default cyan
            600: '#0284c7',  // Darker cyan
            700: '#0369a1',  // Dark cyan
            800: '#075985',  // Very dark cyan
            900: '#0c4a6e',  // Darkest cyan
            950: '#082f49'   // Ultra dark cyan
          },
  
          // Custom colors for meeting intelligence features
          meeting: {
            // Transcript colors
            transcript: '#6366f1',    // Indigo for transcript text
            speaker: '#8b5cf6',       // Purple for speaker labels
            timestamp: '#6b7280',     // Gray for timestamps
            
            // Insight colors
            action: '#10b981',        // Emerald for action items
            decision: '#f59e0b',      // Amber for decisions
            sentiment: '#ec4899',     // Pink for sentiment
            summary: '#3b82f6',       // Blue for summaries
            
            // Status colors
            processing: '#f97316',    // Orange for processing
            completed: '#22c55e',     // Green for completed
            failed: '#ef4444',        // Red for failed
            pending: '#6b7280'        // Gray for pending
          }
        },
  
        // Custom font families
        fontFamily: {
          sans: [
            'Inter',           // Modern, readable sans-serif
            'ui-sans-serif',
            'system-ui',
            '-apple-system',
            'BlinkMacSystemFont',
            'Segoe UI',
            'Roboto',
            'Helvetica Neue',
            'Arial',
            'Noto Sans',
            'sans-serif'
          ],
          mono: [
            'JetBrains Mono',  // For code and timestamps
            'SF Mono',
            'Monaco',
            'Inconsolata',
            'Roboto Mono',
            'source-code-pro',
            'Menlo',
            'monospace'
          ]
        },
  
        // Custom spacing scale
        spacing: {
          '18': '4.5rem',      // 72px
          '22': '5.5rem',      // 88px
          '26': '6.5rem',      // 104px
          '30': '7.5rem',      // 120px
          '34': '8.5rem',      // 136px
          '38': '9.5rem',      // 152px
          '42': '10.5rem',     // 168px
          '46': '11.5rem',     // 184px
          '50': '12.5rem',     // 200px
          '128': '32rem',      // 512px
          '144': '36rem'       // 576px
        },
  
        // Custom border radius
        borderRadius: {
          'xl': '0.75rem',     // 12px
          '2xl': '1rem',       // 16px
          '3xl': '1.5rem',     // 24px
          '4xl': '2rem'        // 32px
        },
  
        // Custom animations for real-time updates
        animation: {
          // Pulse animation for processing states
          'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
          
          // Fade in animation for new content
          'fade-in': 'fadeIn 0.5s ease-in-out',
          'fade-in-up': 'fadeInUp 0.5s ease-out',
          'fade-in-down': 'fadeInDown 0.5s ease-out',
          
          // Slide animations for panels
          'slide-in-right': 'slideInRight 0.3s ease-out',
          'slide-in-left': 'slideInLeft 0.3s ease-out',
          'slide-out-right': 'slideOutRight 0.3s ease-in',
          'slide-out-left': 'slideOutLeft 0.3s ease-in',
          
          // Bounce animation for notifications
          'bounce-in': 'bounceIn 0.6s ease-out',
          
          // Wiggle animation for interactive elements
          'wiggle': 'wiggle 1s ease-in-out infinite',
          
          // Progress bar animation
          'progress': 'progress 2s ease-in-out',
          
          // Typing indicator animation
          'typing': 'typing 1.5s ease-in-out infinite',
          
          // Real-time update indicator
          'live-update': 'liveUpdate 2s ease-in-out infinite'
        },
  
        // Custom keyframes for animations
        keyframes: {
          fadeIn: {
            '0%': { opacity: '0' },
            '100%': { opacity: '1' }
          },
          fadeInUp: {
            '0%': { 
              opacity: '0',
              transform: 'translateY(20px)'
            },
            '100%': { 
              opacity: '1',
              transform: 'translateY(0)'
            }
          },
          fadeInDown: {
            '0%': { 
              opacity: '0',
              transform: 'translateY(-20px)'
            },
            '100%': { 
              opacity: '1',
              transform: 'translateY(0)'
            }
          },
          slideInRight: {
            '0%': { 
              transform: 'translateX(100%)',
              opacity: '0'
            },
            '100%': { 
              transform: 'translateX(0)',
              opacity: '1'
            }
          },
          slideInLeft: {
            '0%': { 
              transform: 'translateX(-100%)',
              opacity: '0'
            },
            '100%': { 
              transform: 'translateX(0)',
              opacity: '1'
            }
          },
          slideOutRight: {
            '0%': { 
              transform: 'translateX(0)',
              opacity: '1'
            },
            '100%': { 
              transform: 'translateX(100%)',
              opacity: '0'
            }
          },
          slideOutLeft: {
            '0%': { 
              transform: 'translateX(0)',
              opacity: '1'
            },
            '100%': { 
              transform: 'translateX(-100%)',
              opacity: '0'
            }
          },
          bounceIn: {
            '0%': {
              transform: 'scale(0.3)',
              opacity: '0'
            },
            '50%': {
              transform: 'scale(1.05)'
            },
            '70%': {
              transform: 'scale(0.9)'
            },
            '100%': {
              transform: 'scale(1)',
              opacity: '1'
            }
          },
          wiggle: {
            '0%, 100%': { 
              transform: 'rotate(-3deg)' 
            },
            '50%': { 
              transform: 'rotate(3deg)' 
            }
          },
          progress: {
            '0%': { 
              width: '0%' 
            },
            '100%': { 
              width: '100%' 
            }
          },
          typing: {
            '0%, 60%, 100%': {
              transform: 'translateY(0px)',
              opacity: '0.4'
            },
            '30%': {
              transform: 'translateY(-10px)',
              opacity: '1'
            }
          },
          liveUpdate: {
            '0%, 100%': {
              transform: 'scale(1)',
              opacity: '1'
            },
            '50%': {
              transform: 'scale(1.05)',
              opacity: '0.8'
            }
          }
        },
  
        // Custom box shadows
        boxShadow: {
          'soft': '0 2px 15px 0 rgba(0, 0, 0, 0.1)',
          'medium': '0 4px 25px 0 rgba(0, 0, 0, 0.15)',
          'strong': '0 10px 40px 0 rgba(0, 0, 0, 0.2)',
          'inner-soft': 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)',
          
          // Colored shadows for different states
          'primary': '0 4px 20px 0 rgba(59, 130, 246, 0.3)',
          'success': '0 4px 20px 0 rgba(34, 197, 94, 0.3)',
          'warning': '0 4px 20px 0 rgba(245, 158, 11, 0.3)',
          'error': '0 4px 20px 0 rgba(239, 68, 68, 0.3)'
        },
  
        // Custom backdrop blur
        backdropBlur: {
          xs: '2px'
        },
  
        // Custom z-index scale
        zIndex: {
          '60': '60',
          '70': '70',
          '80': '80',
          '90': '90',
          '100': '100'
        },
  
        // Custom breakpoints for responsive design
        screens: {
          'xs': '475px',       // Extra small devices
          'sm': '640px',       // Small devices (default)
          'md': '768px',       // Medium devices (default)
          'lg': '1024px',      // Large devices (default)
          'xl': '1280px',      // Extra large devices (default)
          '2xl': '1536px',     // 2X large devices (default)
          '3xl': '1920px',     // 3X large devices (custom)
          
          // Height-based breakpoints for vertical layouts
          'h-sm': { 'raw': '(min-height: 640px)' },
          'h-md': { 'raw': '(min-height: 768px)' },
          'h-lg': { 'raw': '(min-height: 1024px)' }
        }
      }
    },
  
    // Tailwind CSS plugins
    plugins: [
      // Forms plugin for better form styling
      require('@tailwindcss/forms')({
        strategy: 'class' // Use class-based form styling
      }),
  
      // Typography plugin for rich text content
      require('@tailwindcss/typography'),
  
      // Line clamp plugin for text truncation
      require('@tailwindcss/line-clamp'),
  
      // Aspect ratio plugin for maintaining aspect ratios
      require('@tailwindcss/aspect-ratio'),
  
      // Custom plugin for meeting intelligence specific utilities
      function({ addUtilities, addComponents, theme }) {
        // Custom utilities for meeting intelligence UI
        const newUtilities = {
          // Transcript specific utilities
          '.transcript-line': {
            '@apply border-l-4 border-gray-200 pl-4 py-2 hover:border-primary-300 transition-colors duration-200': {}
          },
          '.transcript-speaker': {
            '@apply font-semibold text-meeting-speaker text-sm uppercase tracking-wide': {}
          },
          '.transcript-timestamp': {
            '@apply text-meeting-timestamp text-xs font-mono': {}
          },
          '.transcript-text': {
            '@apply text-gray-700 dark:text-gray-300 leading-relaxed': {}
          },
  
          // Action item utilities
          '.action-item-card': {
            '@apply bg-white dark:bg-gray-800 rounded-lg shadow-soft border border-gray-200 dark:border-gray-700 p-4 hover:shadow-medium transition-shadow duration-200': {}
          },
          '.action-priority-urgent': {
            '@apply border-l-4 border-error-500 bg-error-50 dark:bg-error-900/20': {}
          },
          '.action-priority-high': {
            '@apply border-l-4 border-warning-500 bg-warning-50 dark:bg-warning-900/20': {}
          },
          '.action-priority-medium': {
            '@apply border-l-4 border-info-500 bg-info-50 dark:bg-info-900/20': {}
          },
          '.action-priority-low': {
            '@apply border-l-4 border-gray-400 bg-gray-50 dark:bg-gray-800': {}
          },
  
          // Status indicators
          '.status-processing': {
            '@apply bg-meeting-processing text-white': {}
          },
          '.status-completed': {
            '@apply bg-meeting-completed text-white': {}
          },
          '.status-failed': {
            '@apply bg-meeting-failed text-white': {}
          },
          '.status-pending': {
            '@apply bg-meeting-pending text-white': {}
          },
  
          // Animation utilities
          '.animate-fade-in-fast': {
            'animation': 'fadeIn 0.2s ease-out'
          },
          '.animate-slide-up': {
            'animation': 'fadeInUp 0.4s ease-out'
          },
  
          // Glass morphism effect
          '.glass': {
            '@apply bg-white/10 backdrop-blur-md border border-white/20': {}
          },
          '.glass-dark': {
            '@apply bg-black/10 backdrop-blur-md border border-black/20': {}
          }
        }
  
        // Custom components for common patterns
        const newComponents = {
          // Button variants
          '.btn-primary': {
            '@apply bg-primary-600 hover:bg-primary-700 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200 shadow-sm hover:shadow-primary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2': {}
          },
          '.btn-secondary': {
            '@apply bg-secondary-100 hover:bg-secondary-200 text-secondary-700 font-medium py-2 px-4 rounded-lg transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-secondary-500 focus:ring-offset-2': {}
          },
          '.btn-success': {
            '@apply bg-success-600 hover:bg-success-700 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200 shadow-sm hover:shadow-success': {}
          },
          '.btn-danger': {
            '@apply bg-error-600 hover:bg-error-700 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200 shadow-sm hover:shadow-error': {}
          },
  
          // Card components
          '.card': {
            '@apply bg-white dark:bg-gray-800 rounded-xl shadow-soft border border-gray-200 dark:border-gray-700': {}
          },
          '.card-header': {
            '@apply px-6 py-4 border-b border-gray-200 dark:border-gray-700': {}
          },
          '.card-body': {
            '@apply px-6 py-4': {}
          },
          '.card-footer': {
            '@apply px-6 py-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-700/50 rounded-b-xl': {}
          },
  
          // Loading states
          '.skeleton': {
            '@apply animate-pulse bg-gray-200 dark:bg-gray-700 rounded': {}
          },
          '.skeleton-text': {
            '@apply skeleton h-4 w-full mb-2': {}
          },
          '.skeleton-circle': {
            '@apply skeleton rounded-full': {}
          },
  
          // Navigation components
          '.nav-link': {
            '@apply flex items-center px-3 py-2 text-sm font-medium rounded-md text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors duration-200': {}
          },
          '.nav-link-active': {
            '@apply nav-link bg-primary-50 dark:bg-primary-900/50 text-primary-700 dark:text-primary-300 border-r-2 border-primary-500': {}
          }
        }
  
        // Add the utilities and components
        addUtilities(newUtilities)
        addComponents(newComponents)
      }
    ],
  
    // Safelist - classes that should never be purged
    safelist: [
      // Dynamic classes that might be generated by Angular
      'bg-success-100',
      'bg-warning-100', 
      'bg-error-100',
      'bg-info-100',
      'text-success-700',
      'text-warning-700',
      'text-error-700',
      'text-info-700',
      
      // Animation classes
      'animate-pulse',
      'animate-bounce',
      'animate-fade-in',
      'animate-slide-in-right',
      'animate-slide-in-left',
      
      // Priority classes
      'action-priority-urgent',
      'action-priority-high', 
      'action-priority-medium',
      'action-priority-low',
      
      // Status classes
      'status-processing',
      'status-completed',
      'status-failed',
      'status-pending'
    ]
  }
  