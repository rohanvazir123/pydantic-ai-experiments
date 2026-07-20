// Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
// See LICENSE file in the project root for details.

import { test, expect } from '@playwright/test'

async function login(page: any) {
  await page.goto('/login')
  // Race: either session restores and redirects to /chat, or the login form appears.
  const which = await Promise.race([
    page.waitForURL('**/chat', { timeout: 15_000 }).then(() => 'redirected'),
    page.waitForSelector('button[type="submit"]', { timeout: 15_000 }).then(() => 'form'),
  ])
  if (which === 'form') {
    await page.click('button[type="submit"]')
    await page.waitForURL('**/chat', { timeout: 15_000 })
  }
}

test('logs page shows entries after API activity', async ({ page }) => {
  await login(page)

  // Generate some API activity by hitting health a few times via the browser
  // (the API is already receiving traffic from login + chat navigation)

  // Navigate to logs in a new tab (matching the app's open-in-new-tab behaviour)
  const logsPage = await page.context().newPage()
  await logsPage.goto('/logs')

  // Wait for page to settle
  await logsPage.waitForLoadState('networkidle')

  // The table should render
  await expect(logsPage.locator('table')).toBeVisible({ timeout: 10_000 })

  // Should NOT show the empty state — real log entries should be present
  const emptyState = logsPage.getByText('No logs yet')
  const hasLogs = logsPage.locator('tbody tr').first()

  // Either we have rows or we check the empty state is gone
  const rowCount = await logsPage.locator('tbody tr').count()

  // A fresh server might briefly have 0 rows if the filter hides them all;
  // make sure at least the page rendered without error
  if (rowCount === 0) {
    // Toggle DEBUG on to reveal all levels
    await logsPage.getByRole('button', { name: 'DEBUG' }).click()
    await logsPage.waitForTimeout(500)
  }

  const finalRowCount = await logsPage.locator('tbody tr').count()
  expect(finalRowCount, 'Expected log entries to be visible in the logs page').toBeGreaterThan(0)
})

test('logs page shows empty state text when no logs', async ({ page, context }) => {
  // This test verifies the empty state message is correct — run in isolation
  // by pointing at a fake logs endpoint. We just check the text itself.
  await page.route('**/api/v2/logs**', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ data: [], error: null, request_id: 'test' }),
    })
  })

  await page.route('**/api/v2/auth/**', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ data: { access_token: 'stub' }, error: null, request_id: 'test' }),
    })
  })

  await page.goto('/logs')
  await page.waitForLoadState('networkidle')

  await expect(page.getByText('No logs yet')).toBeVisible({ timeout: 10_000 })
})
