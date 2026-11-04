"""Exercise the frozen-result UI on desktop and mobile with optional Playwright."""
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8765/blog/')
    parser.add_argument('--screenshots', default='/tmp/gepa-wsp-blog-check')
    args = parser.parse_args()
    out = Path(args.screenshots)
    out.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1050}, device_scale_factor=1)
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(args.url, wait_until='networkidle')
        assert page.locator('h1').count() == 1
        assert page.locator('#all-results tr').count() == 54
        assert page.locator('#reliability-table tr').count() == 15
        assert page.locator('#budget-accuracy').inner_text().replace('\n', '') == '87.43%'
        page.screenshot(path=str(out/'desktop-hero.png'))
        page.locator('#results').screenshot(path=str(out/'desktop-budget.png'))
        for task in ['mnist', 'fashion', 'sst2']:
            page.locator(f'[data-budget-task="{task}"]').click()
            for index, budget in enumerate([0, 5, 10, 30, 60]):
                page.locator('#budget-slider').fill(str(index))
                assert page.locator('#budget-output').inner_text() == f'{budget} calls'
            page.locator(f'[data-model-task="{task}"]').click()
            assert page.locator('.model-card').count() == 3
        page.locator('#show-seeds').uncheck()
        assert page.locator('#budget-chart svg').count() == 1
        for model in ['deepseek', 'minimax', 'nemotron']:
            page.locator(f'[data-diff-model="{model}"]').click()
            for seed in ['0', '1', '2']:
                page.select_option('#diff-seed', seed)
                assert page.locator('#sample-grids img').count() == 3
                assert page.locator('#sample-grids img').evaluate_all('(imgs) => imgs.every(i => i.complete && i.naturalWidth > 0)')
        page.select_option('#filter-group', 'models')
        page.select_option('#filter-task', 'fashion')
        page.select_option('#filter-model', 'minimax')
        assert page.locator('#all-results tr:visible').count() == 3
        assert page.locator('#row-count').inner_text() == '3 rows'
        page.select_option('#filter-group', 'diffusion')
        assert page.locator('#empty-results').is_visible()
        for key in ['group', 'task', 'model']:
            page.select_option('#filter-'+key, 'all')
        assert page.locator('#all-results tr:visible').count() == 54
        page.locator('[data-diff-model="deepseek"]').click()
        page.select_option('#diff-seed', '0')
        page.locator('#diffusion').screenshot(path=str(out/'desktop-diffusion.png'))
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        for width in [390, 360]:
            page.set_viewport_size({'width': width, 'height': 844})
            page.goto(args.url, wait_until='networkidle')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), f'Overflow at {width}px'
            assert page.locator('#budget-accuracy').inner_text().replace('\n', '') == '87.43%'
            if width == 390:
                page.screenshot(path=str(out/'mobile-hero.png'))
                page.locator('#results').screenshot(path=str(out/'mobile-budget.png'))
                page.locator('#diffusion').screenshot(path=str(out/'mobile-diffusion.png'))
        assert not errors, errors
        # No remote assets: external references are links, not resource loads.
        resources = page.evaluate('performance.getEntriesByType("resource").map(r=>r.name)')
        assert not any(url.startswith('http') and not url.startswith('http://127.0.0.1:') for url in resources)
        browser.close()
    print('PASS: desktop and 390/360px mobile; all task/budget/model/seed controls; filters; local image loading; no JS errors, overflow or remote resources.')


if __name__ == '__main__':
    main()
