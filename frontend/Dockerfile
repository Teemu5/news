# Use nginx to serve static content
FROM nginx:alpine

# Remove default nginx static assets
#RUN rm -rf /usr/share/nginx/html/*

# Copy static assets over
COPY index.html /usr/share/nginx/html/index.html
COPY favicon.ico /usr/share/nginx/html/favicon.ico

RUN touch /usr/share/nginx/html/wpad.dat

# Expose port 80
EXPOSE 80

# Start nginx and keep it running in the foreground
CMD ["nginx", "-g", "daemon off;"]