FROM alpine:3.19

ENV PATH="/usr/local/bin:$PATH"

RUN apk add --no-cache bash curl ca-certificates docker-cli tar openssl && \
    update-ca-certificates

# Install Supabase CLI using official install script (auto-detects arch)
RUN curl -fsSL https://raw.githubusercontent.com/supabase/cli/main/install.sh \
    | sh -s -- -b /usr/local/bin && \
    supabase --version

WORKDIR /workspace

CMD ["sh", "-lc", "supabase --help"]
