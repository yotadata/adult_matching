FROM alpine:3.20

RUN apk add --no-cache bash docker-cli docker-cli-compose

WORKDIR /workspace

COPY reset.sh /entry.sh
RUN chmod +x /entry.sh

ENTRYPOINT ["/entry.sh"]
