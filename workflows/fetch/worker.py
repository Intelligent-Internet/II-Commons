from lib.config import GlobalConfig
from lib.dataset import init
from lib.gcs import download_file
from lib.s3 import exists, get_address_by_key, upload_file
from lib.utilitas import download, json_dumps, sha256, get_file_type
import os
import sys
import tempfile
import time

BATCH_SIZE = 1
last_item, limit, i = 0, 0, 0
dataset_name = None
ds = None
buffer = []


def get_unprocessed(limit=10, offset=0, mod_by=None, mod_remain=None):
    where_conditions = ['origin_storage_id = %s']
    params = ['']
    if mod_by is not None and mod_remain is not None:
        where_conditions.append('id %% %s = %s')
        params.extend([mod_by, mod_remain])
    resp = ds.query(f'SELECT * FROM {ds.get_table_name()}'
                 + ' WHERE ' + ' AND '.join(where_conditions)
                 + ' AND id > %s ORDER BY id ASC LIMIT %s',
                 tuple(params + [offset, limit]))
    res = []
    for item in resp:
        nItem = item.copy()
        for field in item.keys():
            if field.startswith('vector'):
                nItem.pop(field, None)
        res.append(nItem)
    return res


def trigger(force=False):
    global buffer
    if force or len(buffer) >= BATCH_SIZE:
        if GlobalConfig.DRYRUN:
            print(f"Dryrun: {buffer}")
        else:
            download_data({'meta_items': buffer})
        buffer = []


def sleep(seconds=1):
    print(f'⏰ Sleeping for {seconds} second(s) to avoid rate limit...')
    time.sleep(seconds)


def download_data(args) -> dict:
    meta_items = args['meta_items'] if type(args['meta_items']) == list \
        else [args['meta_items']]
    task_hash = sha256(json_dumps(meta_items))
    temp_path = tempfile.TemporaryDirectory(suffix=f'-{task_hash}')
    results = []
    for meta in meta_items:
        snapshot = ds.snapshot(meta)
        print(f'✨ Processing item: {snapshot}')
        s3_key = ds.get_s3_key(meta)
        match dataset_name:
            case 'arxiv':
                subfix = 'pdf'
            case _:
                subfix = 'jpg'
        filename = os.path.join(temp_path.name, f"{sha256(meta['url'])}.{subfix}")
        try:
            if exists(s3_key):
                meta['origin_storage_id'] = get_address_by_key(s3_key)
                print(f"Skipping download: {meta['origin_storage_id']}")
            else:
                match dataset_name:
                    case 'arxiv':
                        arr_pid = meta['paper_id'].split('/')
                        if len(arr_pid) == 1:
                            gcs_url = f"gs://arxiv-dataset/arxiv/arxiv/pdf/{meta['paper_id'].split('.')[0]}/{meta['paper_id']}{meta['versions'][-1]}.pdf"
                        elif len(arr_pid) == 2:
                            gcs_url = f"gs://arxiv-dataset/arxiv/{arr_pid[0]}/pdf/{arr_pid[1][:4]}/{arr_pid[1]}{meta['versions'][-1]}.pdf"
                        else:
                            print(meta)
                            raise ValueError('Invalid paper_id.')
                        try:
                            # hack:
                            os.environ['GCS_BUCKET'] = "arxiv-dataset"
                            download_file(gcs_url, filename)
                            print(f"Downloaded {gcs_url} to: {filename}")
                        except Exception as e:
                            print(
                                f"Fownload failed from GCS: {gcs_url}, try direct download..."
                            )
                            download(meta['url'], filename)
                            print(f"Downloaded {meta['url']} to: {filename}")
                        if get_file_type(filename) != 'PDF':
                            raise ValueError('Unexpected file type.')
                        meta['origin_storage_id'] = upload_file(filename, s3_key)
                        print(f"Uploaded to S3: {meta['origin_storage_id']}")
                    case _:
                        download(meta['url'], filename)
                        print(f"Downloaded {meta['url']} to: {filename}")
                        meta['origin_storage_id'] = upload_file(filename, s3_key)
                        print(f"Uploaded to S3: {meta['origin_storage_id']}")
            ds.update_by_id(meta['id'], {
                'origin_storage_id': meta['origin_storage_id'],
            })
            print(f'🔥 Inserted meta: {snapshot}')
        except Exception as e:
            print(f'❌ Error handling {snapshot}: {e}')
            continue
        results.append(meta)
    print('👌 Done!')
    return {'meta_items': results}


def run(name):
    global buffer, last_item, dataset_name, ds
    i = 0
    dataset_name = name
    try:
        ds = init(dataset_name)
    except Exception as e:
        print(f"❌ Unable to init src-dataset: {dataset_name}. Error: {e}")
        sys.exit(1)
    meta_items = get_unprocessed()
    while len(meta_items) > 0:
        should_break = False
        for meta in meta_items:
            i += 1
            last_item = meta['id']
            meta_snapshot = ds.snapshot(meta)
            print(f"Processing row {i} - {last_item}: {meta_snapshot}")
            # print(meta)
            buffer.append(meta)
            trigger()
            if i >= limit > 0:
                should_break = True
                break
        if should_break:
            break
        meta_items = get_unprocessed()
    trigger(force=True)
    print('All Done!')


__all__ = [
    'run'
]
