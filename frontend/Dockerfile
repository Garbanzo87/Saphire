# ---- build stage -----------------------------------------------------------
FROM node:22-alpine AS build
WORKDIR /app

# NEXT_PUBLIC_* values are inlined into the static bundle at build time.
ARG NEXT_PUBLIC_SAPHIRE_API=http://localhost:8000
ARG NEXT_PUBLIC_SAPHIRE_API_KEY=dev-key
ARG NEXT_PUBLIC_SAPHIRE_PROJECT=default
ENV NEXT_PUBLIC_SAPHIRE_API=$NEXT_PUBLIC_SAPHIRE_API \
    NEXT_PUBLIC_SAPHIRE_API_KEY=$NEXT_PUBLIC_SAPHIRE_API_KEY \
    NEXT_PUBLIC_SAPHIRE_PROJECT=$NEXT_PUBLIC_SAPHIRE_PROJECT \
    NEXT_TELEMETRY_DISABLED=1

COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build   # output: 'export' -> ./out

# ---- serve stage -----------------------------------------------------------
FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/out /usr/share/nginx/html
EXPOSE 3000
CMD ["nginx", "-g", "daemon off;"]
