"""Read-only local artist search endpoints for the ComfyUI extension."""
import asyncio
import csv
import io
import json
from aiohttp import web
from safetensors import SafetensorError
from .catalog import artist_index


def register_routes(routes, bank_paths):
    @routes.post('/kaloscope-adapter/migrate-workflow')
    async def migrate(request):
        from .workflows import migrate_workflow
        try:
            return web.json_response(migrate_workflow(await request.json()))
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            return web.json_response({'error': 'Invalid workflow graph.'}, status=400)

    def index_for(request):
        name = request.query.get('bank', '')
        if not name:
            return artist_index()
        path = bank_paths().get(name)
        if path is None:
            raise ValueError('Selected artist bank is not installed. Refresh the bank list.')
        return artist_index(path)

    @routes.get('/kaloscope-adapter/artists')
    async def search(request):
        try:
            limit = int(request.query.get('limit', '40'))
            query = request.query.get('q', '')
            if not 1 <= limit <= 100 or len(query) > 160:
                raise ValueError('Use a limit of 1–100 and at most 160 search characters.')
            def run():
                return index_for(request).search(query, limit)
            result = await asyncio.to_thread(run)
            return web.json_response(result)
        except (ValueError, FileNotFoundError, KeyError, SafetensorError) as exc:
            return web.json_response({'error': str(exc)}, status=400)

    @routes.get('/kaloscope-adapter/artists/catalog.csv')
    async def download(request):
        try:
            def render():
                index = index_for(request)
                out = io.StringIO(newline='')
                writer = csv.writer(out)
                writer.writerow(['name', 'aliases', 'families', 'concentration'])
                for row in index.rows:
                    writer.writerow([row['name'], json.dumps(row['aliases'], ensure_ascii=False), row['families'], row['concentration']])
                return out.getvalue().encode('utf-8-sig')
            content = await asyncio.to_thread(render)
            return web.Response(body=content, content_type='text/csv', charset='utf-8',
                                headers={'Content-Disposition': 'attachment; filename="kaloscope_artists.csv"'})
        except (ValueError, FileNotFoundError, KeyError, SafetensorError) as exc:
            return web.json_response({'error': str(exc)}, status=400)
