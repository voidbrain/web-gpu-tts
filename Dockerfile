# Stage 1: Build the Angular application
FROM node:20-alpine as builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm install

COPY . .
RUN npm run build -- --configuration production

# Debugging step: List contents of the built application
RUN ls -R /app/dist/nils-whisper-web-gpu-app/browser

# Stage 2: Serve the application with Nginx
FROM nginx:alpine

COPY --from=builder /app/dist/nils-whisper-web-gpu-app/browser /usr/share/nginx/html

EXPOSE 4201

CMD ["nginx", "-g", "daemon off;"]
