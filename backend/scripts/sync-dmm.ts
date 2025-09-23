#!/usr/bin/env -S deno run --allow-net --allow-env

/**
 * DMM APIデータ同期スクリプト
 * 
 * 使用方法:
 * deno run --allow-net --allow-env scripts/sync-dmm.ts
 * deno run --allow-net --allow-env scripts/sync-dmm.ts --page=1 --limit=50
 */

interface SyncOptions {
  page?: number;
  limit?: number;
  apiId?: string;
  affiliateId?: string;
}

async function syncDmmData(options: SyncOptions = {}) {
  const supabaseUrl = Deno.env.get('NEXT_PUBLIC_SUPABASE_URL') || 'http://127.0.0.1:54321';
  const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
  
  if (!serviceRoleKey) {
    console.error('SUPABASE_SERVICE_ROLE_KEY is required');
    Deno.exit(1);
  }

  const url = `${supabaseUrl}/functions/v1/dmm_sync`;
  
  const payload = {
    page: options.page || 1,
    limit: options.limit || 100,
    ...(options.apiId && { api_id: options.apiId }),
    ...(options.affiliateId && { affiliate_id: options.affiliateId })
  };

  console.log(`Starting DMM sync...`);
  console.log(`URL: ${url}`);
  console.log(`Payload:`, payload);

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${serviceRoleKey}`,
      },
      body: JSON.stringify(payload),
    });

    const result = await response.json();

    if (!response.ok) {
      console.error('Sync failed:', result);
      Deno.exit(1);
    }

    console.log('Sync completed successfully:');
    console.log(`- Inserted: ${result.inserted_count} videos`);
    console.log(`- Total fetched: ${result.total_fetched}`);
    console.log(`- Total available: ${result.total_available}`);
    console.log(`- Errors: ${result.errors?.length || 0}`);
    
    if (result.errors?.length > 0) {
      console.log('Errors:', result.errors);
    }

    if (result.next_page) {
      console.log(`Next page available: ${result.next_page}`);
    }

  } catch (error) {
    console.error('Network error:', error.message);
    Deno.exit(1);
  }
}

// コマンドライン引数の解析
const args = Deno.args;
const options: SyncOptions = {};

for (const arg of args) {
  const [key, value] = arg.split('=');
  if (key === '--page') options.page = parseInt(value);
  if (key === '--limit') options.limit = parseInt(value);
  if (key === '--api-id') options.apiId = value;
  if (key === '--affiliate-id') options.affiliateId = value;
}

// ヘルプ表示
if (args.includes('--help') || args.includes('-h')) {
  console.log(`
DMM API Data Sync Script

Usage:
  deno run --allow-net --allow-env scripts/sync-dmm.ts [options]

Options:
  --page=N          Page number to fetch (default: 1)
  --limit=N         Items per page (default: 100)
  --api-id=ID       DMM API ID (optional, uses env var if not set)
  --affiliate-id=ID DMM Affiliate ID (optional, uses env var if not set)
  --help, -h        Show this help message

Environment Variables:
  NEXT_PUBLIC_SUPABASE_URL     Supabase project URL
  SUPABASE_SERVICE_ROLE_KEY    Supabase service role key
  DMM_API_ID                   DMM API ID
  DMM_AFFILIATE_ID             DMM Affiliate ID
`);
  Deno.exit(0);
}

// 実行
await syncDmmData(options);