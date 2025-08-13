# Dockerfile-react (corrigé)

# --- ÉTAPE 1: Build ---
FROM node:20-alpine as builder

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# --- ÉTAPE 2: Production ---
FROM nginx:alpine


COPY --from=builder /app/dist /usr/share/nginx/html

COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]