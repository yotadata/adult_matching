const { createClient } = require('@supabase/supabase-js');

const supabaseUrl = 'http://127.0.0.1:54321';
const serviceRoleKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU';
const supabase = createClient(supabaseUrl, serviceRoleKey);

async function quickCheck() {
  try {
    console.log('🔍 Quick Database Check');
    console.log('======================');
    
    // Count total DMM videos
    const { data: count, error } = await supabase
      .from('videos')
      .select('count', { count: 'exact' })
      .eq('source', 'dmm');
    
    if (error) throw error;
    
    console.log(`📊 Total DMM videos: ${count[0].count}`);
    
    // Check created today
    const today = new Date().toISOString().split('T')[0];
    const { data: todayCount, error: todayError } = await supabase
      .from('videos')
      .select('count', { count: 'exact' })
      .eq('source', 'dmm')
      .gte('created_at', `${today}T00:00:00.000Z`)
      .lte('created_at', `${today}T23:59:59.999Z`);
    
    if (todayError) throw todayError;
    
    console.log(`📅 Created today: ${todayCount[0].count}`);
    
    // Get date range
    const { data: dateRange, error: dateError } = await supabase
      .from('videos')
      .select('created_at')
      .eq('source', 'dmm')
      .order('created_at', { ascending: false })
      .limit(1);
    
    if (dateError) throw dateError;
    
    console.log(`🕐 Latest record: ${dateRange[0].created_at}`);
    
  } catch (error) {
    console.error('❌ Error:', error.message);
  }
}

quickCheck();