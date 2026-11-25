"""Read-only access to the shared game master catalogue."""

import json
import zlib
import logging
import sqlite3
from pathlib import Path
from contextlib import closing

if __package__:
    from .utils.runtime_paths import resolve_runtime_path
else:
    from utils.runtime_paths import resolve_runtime_path

SCHEMA_VERSION = 1
logger = logging.getLogger(__name__)


def master_catalog_path():
    return resolve_runtime_path(Path('.local/master-data/catalog.sqlite'), Path('data/master/catalog.sqlite'))


class MasterCatalog:
    """Preserve source order and return every matching row, including duplicate IDs."""

    def __init__(self, path):
        self.path = Path(path).resolve()

    def __enter__(self):
        self.db = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)
        with closing(self.db.cursor()) as cursor:
            version, = cursor.execute('PRAGMA user_version').fetchone()
        if version != SCHEMA_VERSION:
            self.db.close()
            raise ValueError(f'unsupported master catalogue schema: {version}')
        return self

    def __exit__(self, *_):
        self.db.close()

    def tables(self):
        with closing(self.db.cursor()) as cursor:
            cursor.row_factory = sqlite3.Row
            return tuple(dict(row) for row in cursor.execute('SELECT * FROM data_tables ORDER BY source, name'))

    def table_info(self, table, *, source='diff'):
        with closing(self.db.cursor()) as cursor:
            cursor.row_factory = sqlite3.Row
            row = cursor.execute(
                'SELECT * FROM data_tables WHERE source = ? AND name = ?', (source, table),
            ).fetchone()
        return dict(row) if row is not None else None

    def rows(self, table, *, source='diff', key=None, form=None, name=None, pattern=None):
        params = [source, table]
        constraints = ['source = ?', 'table_name = ?']
        for column, value in (
            ('row_id = ?', key), ('form = ?', form), ('row_name = ?', name), ('row_id GLOB ?', pattern),
        ):
            if value is not None:
                constraints.append(column)
                params.append(value)
        query = 'SELECT payload FROM records WHERE ' + ' AND '.join(constraints) + ' ORDER BY ordinal'
        with closing(self.db.cursor()) as cursor:
            for payload, in cursor.execute(query, params):
                yield json.loads(zlib.decompress(payload))


def shared_master_rows(source, tables, *, revision=None, digests=None, pattern=None):
    """Load a matching view for an existing consumer with its own bundled data."""
    path = master_catalog_path()
    if not path.is_file():
        return None
    try:
        with MasterCatalog(path) as catalog:
            for name in tables:
                info = catalog.table_info(name, source=source)
                if (info is None or (revision is not None and info['revision'] != revision)
                        or (digests is not None and info['sha256'] != digests[name])):
                    return None
            return {name: list(catalog.rows(name, source=source, pattern=pattern)) for name in tables}
    except (sqlite3.Error, ValueError, zlib.error) as error:
        logger.warning('Using existing bundled master data: %s', error)
        return None
