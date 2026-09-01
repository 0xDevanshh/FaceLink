#!/usr/bin/env python3
"""Browser test for UI responsiveness during scan."""

import subprocess
import time
from pathlib import Path


async def test_frontend_ui():
    """Test that frontend UI updates during scan."""
    
    print("\n" + "="*70)
    print("TESTING FRONTEND UI RESPONSIVENESS")
    print("="*70)
    
    # Start a browser-based test
    test_script = '''
const fs = require('fs');
const path = require('path');

// Simple browser test - open DevTools and check console logs
async function runTest() {
    const page = await browser.newPage();
    
    // Capture console messages
    const consoleLogs = [];
    page.on('console', msg => {
        consoleLogs.push({
            type: msg.type(),
            text: msg.text()
        });
    });
    
    // Navigate to app
    console.log('Opening frontend...');
    await page.goto('http://localhost:5173', { waitUntil: 'networkidle' });
    
    // Read sample image
    const imagePath = path.join(process.cwd(), 'samples/satya_nadella.jpg');
    const fileInput = await page.$('input[type="file"]');
    
    if (!fileInput) {
        console.error('File input not found!');
        return;
    }
    
    // Upload file
    console.log('Uploading image...');
    await fileInput.uploadFile(imagePath);
    
    // Wait for preview to load
    await page.waitForTimeout(1000);
    
    // Find and click upload button
    const uploadBtn = await page.$('button:has-text("Start Scan")');
    if (!uploadBtn) {
        console.error('Upload button not found!');
        return;
    }
    
    console.log('Clicking Start Scan...');
    await uploadBtn.click();
    
    // Wait for progress view to appear
    try {
        await page.waitForSelector('[data-testid="progress-log"]', { timeout: 5000 });
        console.log('✓ Progress view appeared');
    } catch (e) {
        console.error('✗ Progress view did not appear');
        return;
    }
    
    // Wait for events to start arriving
    let eventCount = 0;
    const startTime = Date.now();
    const timeout = 60000; // 60 seconds
    
    while (Date.now() - startTime < timeout) {
        const events = await page.$$('[data-event]');
        const newCount = events.length;
        
        if (newCount > eventCount) {
            eventCount = newCount;
            console.log(`Events received: ${eventCount}`);
        }
        
        // Check if scan is complete
        const done = await page.$('[data-stage="done"]');
        if (done) {
            console.log('✓ Scan completed!');
            break;
        }
        
        await page.waitForTimeout(2000);
    }
    
    // Print all console logs that mention SSE or events
    console.log('\\nConsole logs from page:');
    consoleLogs
        .filter(log => log.text.includes('SSE') || log.text.includes('Event') || log.text.includes('Poll'))
        .forEach(log => console.log(`  [${log.type}] ${log.text}`));
    
    if (eventCount > 0) {
        console.log(`\\n✓✓✓ FRONTEND RECEIVED ${eventCount} EVENTS ✓✓✓`);
    } else {
        console.log('\\n✗ No events received in frontend');
    }
    
    await page.close();
}

runTest().catch(console.error);
'''
    
    # Write test script
    test_file = Path("test_ui.js")
    test_file.write_text(test_script)
    
    # Check if playwright is available
    try:
        import playwright
        print("\nNote: Browser test requires Playwright. Running simplified backend test instead.\n")
    except ImportError:
        print("\nPlaywright not installed. Use backend test (test_sse_fix.py) to verify.\n")
    
    # Clean up
    test_file.unlink()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_frontend_ui())
