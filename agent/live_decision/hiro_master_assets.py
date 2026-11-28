"""Present Hiro's asset tables from the packaged simulator snapshot."""

import json
import sqlite3
from hashlib import file_digest
from threading import RLock


def _identifier(name):
    return '"' + name.replace('"', '""') + '"'


class _MasterView(sqlite3.Connection):
    def __init__(self, catalog, decode, cell):
        # CPython 3.12 can reuse a cached statement while another thread is fetching it.
        super().__init__(':memory:', check_same_thread=False, cached_statements=0)
        super().execute('PRAGMA temp_store = MEMORY')
        self.row_factory = sqlite3.Row
        self._catalog = catalog
        self._decode = decode
        self._cell = cell
        self._tables = {name.casefold(): name for name, in catalog.execute('SELECT DISTINCT table_name FROM records')}
        self._loaded = set()
        self._lock = RLock()

    def execute(self, sql, parameters=(), /):
        with self._lock:
            while True:
                try:
                    return super().execute(sql, parameters)
                except sqlite3.OperationalError as error:
                    prefix = 'no such table: '
                    message = str(error)
                    if not message.startswith(prefix):
                        raise
                    requested = message.removeprefix(prefix).removeprefix('main.')
                    table = self._tables.get(requested.casefold())
                    if table is None or table in self._loaded:
                        raise
                    self._read_table(table)

    def _read_table(self, table):
        source = self._catalog.execute(
            'SELECT payload FROM records WHERE table_name = ? ORDER BY ordinal', (table,),
        )
        rows = [self._decode(payload) for (payload,) in source]
        columns = list(dict.fromkeys(key for row in rows for key in row))
        names = ', '.join(map(_identifier, columns))
        placeholders = ', '.join('?' for _ in columns)
        quoted = _identifier(table)
        with self:
            super().execute('BEGIN')
            super().execute(f'CREATE TABLE {quoted} ({names})')
            if 'id' in columns:
                super().execute(f'CREATE INDEX {_identifier(table + "_id")} ON {quoted} (id)')
            super().executemany(f'INSERT INTO {quoted} ({names}) VALUES ({placeholders})', (
                tuple(self._cell(row.get(column)) for column in columns) for row in rows
            ))
        self._loaded.add(table)

    def close(self):
        with self._lock:
            self._catalog.close()
            super().close()


def _open_catalog(runtime, revision):
    tables = runtime / 'vendor/gakumas-player/data/master'
    pointer = json.loads((tables / 'current.json').read_text(encoding='utf-8'))
    if (not isinstance(pointer, dict) or pointer.get('commit') != revision
            or pointer.get('database') != f'{revision}/catalog.sqlite'):
        raise ValueError('Hiro master pointer does not match the pinned snapshot')
    path = tables / revision / 'catalog.sqlite'
    with path.open('rb') as stream:
        if file_digest(stream, 'sha256').hexdigest() != pointer.get('database_sha256'):
            raise ValueError('Hiro master database does not match its pinned digest')
    db = sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True, check_same_thread=False)
    try:
        row = db.execute('SELECT value FROM metadata WHERE key = ?', ('snapshot',)).fetchone()
        snapshot = json.loads(row[0]) if row else {}
        if (not isinstance(snapshot, dict) or snapshot.get('commit') != revision
                or snapshot.get('manifest_sha256') != pointer.get('manifest_sha256')):
            raise ValueError('Hiro master snapshot metadata does not match its pointer')
    except (sqlite3.Error, ValueError, TypeError):
        db.close()
        raise
    return db


def mounted_asset_source():
    """Use Hiro's original asset reader and expose only the queried tables in memory."""
    from hiro import sim  # noqa: F401 -- selects the pinned simulator import root
    from gakumas_exam.catalog import Catalog
    from hiro.defaults.assets import PublicAssetSource
    from hiro.state.master_db import _cell

    from .hiro_runtime import HIRO_MASTER_REVISION, hiro_runtime_root

    class MountedAssetSource(PublicAssetSource):
        def __init__(self):
            super().__init__()
            self._master_lock = RLock()

        def card_art_dir(self):
            if self.prebuilt_card_art_root() is not None:
                return None
            return super().card_art_dir()

        def prebuilt_card_art_root(self):
            runtime = hiro_runtime_root().resolve()
            if self.bundled is None or self.bundled.resolve() != runtime / 'assets/pack':
                return None
            stock = self.bundled / 'cards'
            if stock.is_dir() and any(stock.iterdir()):
                return None
            if not (self.bundled / 'card-index-v1/features.npz').is_file():
                return None
            # The compact bank replaces the 2,290 entries of the pinned stock
            # directory for source selection. PublicAssetSource prefers user
            # art on an equal count.
            custom = self.user / 'cards'
            if custom.is_dir() and sum(1 for _ in custom.iterdir()) >= 2290:
                return None
            return runtime

        def master_db(self):
            with self._master_lock:
                if self._db is None:
                    catalog = _open_catalog(hiro_runtime_root(), HIRO_MASTER_REVISION)
                    try:
                        self._db = _MasterView(catalog, Catalog.decode, _cell)
                    except sqlite3.Error:
                        catalog.close()
                        raise
                return self._db

    return MountedAssetSource()
