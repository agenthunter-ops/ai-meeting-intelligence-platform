# Multi-stage Dockerfile for Angular Frontend
# ==========================================

# Stage 1: Build the application
FROM node:18-alpine AS build

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Improve npm reliability with retries
ENV NPM_CONFIG_FETCH_RETRIES=5 NPM_CONFIG_FETCH_RETRY_MINTIMEOUT=10000 NPM_CONFIG_FETCH_RETRY_MAXTIMEOUT=60000

# Install dependencies
RUN npm install --legacy-peer-deps --prefer-offline --no-audit --only=production

# Copy source code
COPY . .

# Build the application
RUN npm run build:prod

# Stage 2: Development server  
FROM node:20-alpine AS development

WORKDIR /app

# Copy package files
COPY package*.json ./

# Improve npm reliability with retries
ENV NPM_CONFIG_FETCH_RETRIES=5
ENV NPM_CONFIG_FETCH_RETRY_MINTIMEOUT=10000
ENV NPM_CONFIG_FETCH_RETRY_MAXTIMEOUT=60000

# Install all dependencies (including dev dependencies)
RUN npm install --legacy-peer-deps --prefer-offline --no-audit

# Copy source code
COPY . .

# Expose port
EXPOSE 4200

# Start development server
CMD ["npm", "run", "serve:dev"]

# Stage 3: Production server
FROM nginx:alpine AS production

# Copy built application from build stage
COPY --from=build /app/dist/meeting-intelligence-frontend /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"] 