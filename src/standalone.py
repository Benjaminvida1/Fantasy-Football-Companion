"""Wrap the dashboard page fragment (built for the claude.ai artifact viewer, which adds its own document
wrapper) into a standalone HTML document for GitHub Pages.
Usage: python3 standalone.py <in.html> <out.html>   (idempotent: a file that already has a doctype is copied as is)"""
import sys

RESET = ('<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
         'body{margin:0;font:14px system-ui,-apple-system,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style>\n')


def wrap(html):
    if html.lstrip().lower().startswith('<!doctype'):
        return html
    head, body = html.split('</style>', 1)       # title, meta, font links and the page stylesheet end at the first </style>
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + RESET + head + '</style>\n</head>\n<body>\n' + body + '\n</body>\n</html>\n')


if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    out = wrap(open(src).read())
    open(dst, 'w').write(out)
    print(f'wrote {dst} ({len(out) // 1024} KB)')
