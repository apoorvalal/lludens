"""Export the memo for Quomodoc's script-disabled annotation reader."""
import base64
from pathlib import Path
import re
from urllib.parse import urljoin

HERE = Path(__file__).resolve().parent
REPORT_URL = 'http://apoorvas-mac-mini.tail55b852.ts.net/research/lludens-decisions-jev-2026-10-06/'


def export_quomodoc():
    report = HERE / 'report'
    text = (report / 'index.html').read_text()
    text, scripts = re.subn(r'<script>.*?</script>', '', text, flags=re.S)
    assert scripts == 1
    replay = ('<section id="replay"><h2>Interactive replay</h2>'
              '<p>The <a href="' + REPORT_URL + '#replay">private report (Tailscale)</a> '
              'contains the interactive replay of all sixty direct-action matches. '
              'This annotation copy retains the complete results and match-level tables.</p></section>')
    text, sections = re.subn(r'<section id="replay">.*?</section>', replay, text, flags=re.S)
    assert sections == 1
    png = base64.b64encode((report / 'payoffs.png').read_bytes()).decode('ascii')
    assert text.count('src="payoffs.png"') == 1
    text = text.replace('src="payoffs.png"', 'src="data:image/png;base64,' + png + '"')
    text = re.sub(r'href="([^"]+)"',
                  lambda m: 'href="' + (m[1] if m[1].startswith('#') else urljoin(REPORT_URL, m[1])) + '"',
                  text)
    text = text.replace('<section id="download"><h2>Replication</h2>',
                        '<section id="download"><h2>Replication</h2>'
                        '<p>The downloads below are hosted in the private report library and require Tailscale. '
                        'The source PR is on GitHub.</p>')
    assert '<script>' not in text and '<select' not in text and 'src="payoffs.png"' not in text
    output = report / 'quomodoc.html'
    output.write_text(text)
    return output


if __name__ == '__main__':
    print(export_quomodoc())
