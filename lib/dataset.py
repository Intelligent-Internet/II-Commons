from bs4 import BeautifulSoup
from lib.psql import enrich_data, get_dataset, query
from lib.s3 import get_address_by_key
from lib.utilitas import fetch, sha256
import json
import os
import re

S3_PATH = 'ii-datasets'
SQL_PROCESSED = "processed_storage_id != ''"

CC12M_JSONL = {
    'caption': 'caption_llava_short',
    'caption_long': 'caption_llava',
}

I2D_JSON = {
    'origin_hash': 'md5',
}


def split_authors(authors):
    return list(filter(None, [
        re.sub(r'\s+', ' ', x.replace('\n', ' ').strip())
        for x in re.split(r',| and ', authors or '')
    ]))


def extract_image_meta_from_wiki_page(url):
    # demo heml
    # ...
    # <meta property="og:image" content="https://upload.wikimedia.org/wikipedia/commons/f/f8/Cheilopogon_melanurusPCCA20070623-3956B.jpg">
    # <meta property="og:image:width" content="1200">
    # <meta property="og:image:height" content="907">
    # <meta property="og:image" content="https://upload.wikimedia.org/wikipedia/commons/f/f8/Cheilopogon_melanurusPCCA20070623-3956B.jpg">
    # <meta property="og:image:width" content="800">
    # <meta property="og:image:height" content="605">
    # <meta property="og:image:width" content="640">
    # <meta property="og:image:height" content="484">
    # ...
    # Parse HTML content
    try:
        content = fetch(url)
        soup = BeautifulSoup(content, 'html.parser')
        # img = soup.find('img')
        # if img:
        #     return img['src']
        # Find all og:image meta tags and their corresponding width/height
        images = []
        for meta in soup.find_all('meta', property='og:image'):
            img_url = meta.get('content')
            width = meta.find_next('meta', property='og:image:width')
            height = meta.find_next('meta', property='og:image:height')
            if width and height:
                width = int(width.get('content'))
                height = int(height.get('content'))
                area = width * height
                images.append((area, img_url))
        # Get the URL with largest dimensions
        if not images:
            raise Exception("Not able to extract image meta tags!")
        largest_image_url = max(images)[1]
        print(f"Found image in page: {url} => {largest_image_url}")
        return largest_image_url, width, height
    except Exception as e:
        raise Exception(f"Failed to process {url} : {e}")


def map_meta_wikipedia_featured(wiki_meta):
    url, width, height = extract_image_meta_from_wiki_page(
        wiki_meta['image']['url']
    )
    hash = sha256(url)
    return {
        'url': url,
        'caption': wiki_meta['image'].get('caption')
        or wiki_meta['image'].get('image_description')
        or wiki_meta['image'].get('image_parsed_title')
        or wiki_meta['image'].get('image_title')
        or wiki_meta['text'].get('title') or '',
        'caption_long': '',
        'hash': hash,
        'vector': None,
        'origin_hash': os.path.splitext(wiki_meta['image']['filename'])[0],
        'origin_width': width,
        'origin_height': height,
        'origin_storage_id': get_address_by_key(os.path.join(
            RULES['wikipedia_featured']['s3_path'], f"{hash}.jpg"
        )),
        'processed_storage_id': '',
        'processed_width': 0,
        'processed_height': 0,
        'aspect_ratio': width / height if height != 0 else 0,
        'exif': {},
        'meta': {
            'article_id': wiki_meta['text']['id'],
            'article_title': wiki_meta['text']['title'],
            'article_url': wiki_meta['text']['url'],
            # 'article_html': wiki_meta['text']['html'],
            # Exceeds the size limit for the Hatchet API.
            'article_wikitext': wiki_meta['text']['wikitext'],
            'image_description': wiki_meta['image'].get('description', ''),
            'image_heading': wiki_meta['image']['headings'],
            'image_title': wiki_meta['image']['title'],
            'image_parsed_title': wiki_meta['image']['parsed_title'],
            'image_url': wiki_meta['image']['url'],
            'image_features': wiki_meta['image']['features'],
        }
    }


def map_meta_megalith_10m(mg_meta):
    url = mg_meta['url_highres'] or mg_meta['url']
    hash = sha256(url)
    return {
        'url': url,
        'caption': '',
        'caption_long': '',
        'hash': hash,
        'vector': None,
        'origin_hash': '',
        'origin_width': 0,
        'origin_height': 0,
        'origin_storage_id': get_address_by_key(os.path.join(
            RULES['megalith_10m']['s3_path'], f"{hash}.jpg"
        )),
        'processed_storage_id': '',
        'processed_width': 0,
        'processed_height': 0,
        'aspect_ratio': 0,
        'exif': {},
        'meta': {
            'url_lowres': mg_meta['url'],
            'url_source': mg_meta['url_source'],
        }
    }


def map_meta_arxiv(ax_meta):
    if ax_meta.get('abstract_md5') is None:
        hash = sha256(ax_meta['abstract'])
    else:
        hash = ax_meta['abstract_md5']
    resp = {
        'paper_id': ax_meta['id'],
        'submitter': split_authors(ax_meta['submitter']),
        'authors': split_authors(ax_meta['authors']),
        'title': ax_meta['title'],
        'comments': ax_meta['comments'] or '',
        'journal_ref': ax_meta['journal-ref'] or '',
        'doi': ax_meta['doi'] or '',
        'report_no': ax_meta['report-no'] or '',
        'hash': hash,
        'url': f"https://arxiv.org/pdf/{ax_meta['id']}",
        'origin_storage_id': get_address_by_key(os.path.join(
            RULES['arxiv']['s3_path'], f"{hash}.pdf"
        )),
    }
    if ax_meta.get('abstract_md5') is None:
        resp['categories'] = ax_meta['categories'] or []
        resp['versions'] = ax_meta['versions'] or []
    else:
        resp['categories'] = [ax_meta['categories']] if len(
            ax_meta['categories']) > 0 else []
        resp['versions'] = [
            r['version'] for r in ax_meta['versions']
        ] if len(ax_meta['versions']) > 0 else []
        resp['license'] = ax_meta['license'] or ''
        resp['abstract'] = ax_meta['abstract'] or ''
    return resp


RULES = {
    'vintage_450k': {
        'fields': {
            'caption': 'long_caption',
            'origin_hash': 'hash',
            'url': 'image_url',
        },
        's3_path': S3_PATH,
    },
    'cc12m_woman': {
        'fields': CC12M_JSONL,
        's3_path': S3_PATH,
    },
    'cc12m_cleaned': {
        'fields': CC12M_JSONL,
        's3_path': S3_PATH,
    },
    'cc12m': {
        'fields': {},
        's3_path': S3_PATH,
    },
    'pd12m': {
        'fields': {},
        's3_path': S3_PATH,
    },
    'wikipedia_featured': {
        'fields': map_meta_wikipedia_featured,
        's3_path': S3_PATH,
    },
    'megalith_10m': {
        'fields': map_meta_megalith_10m,
        's3_path': S3_PATH,
    },
    'alpha': {
        'fields': {},
        's3_path': S3_PATH,
    },
    'wikipedia_en': {
        'fields': {},
        's3_path': S3_PATH,
    },
    'wikipedia_en_embed': {
        'fields': {},
        's3_path': S3_PATH,
    },
    'arxiv': {
        'fields': map_meta_arxiv,
        's3_path': S3_PATH,
    },
    'ms_marco': {
        'fields': {},
        's3_path': S3_PATH,
    },
    'ms_marco_embed': {
        'fields': {},
        's3_path': S3_PATH,
    },
    'workers': {
        'fields': {},
        's3_path': S3_PATH,
    }
}


def map_field(set_name, meta, field, i2d=False, default=None):
    return meta.get((
        I2D_JSON if i2d else RULES[set_name]
    ).get(field, field)) or default


def init(name, i2d=False):
    if name is None:
        raise Exception('Dataset name is required.')
    if RULES.get(name) is None:
        raise Exception(f'Unsupported dataset: {name}')
    if RULES[name].get('fields') is None and not i2d:
        raise Exception(
            f'Only `img2dataset` mode is supported for this dataset: {name}'
        )
    name = name.lower()
    dataset = get_dataset(name)
    dataset.init()

    def get_s3_key(meta):
        if i2d:
            raise Exception(
                'This function is not supported when using `img2dataset` mode.'
            )
        subfix = 'jpg'
        match name:
            case 'wikipedia_en':
                subfix = 'json'
            case 'arxiv':
                subfix = 'pdf'
        return os.path.join(
            RULES[name].get('s3_path', S3_PATH),
            f"{meta['hash'] if meta.get('hash') else sha256(meta['url'])}.{subfix}"
        )

    def get_s3_key_i2d(origin_meta):
        if not i2d:
            raise Exception(
                'This function is only supported when using `img2dataset` mode.'
            )
        return os.path.join(
            RULES[name].get('s3_path', S3_PATH),
            origin_meta['key'][:5], f"{origin_meta['key']}.jpg"
        )

    def map_meta(meta):

        def mf(field, default=None):
            return map_field(name, meta, field, i2d, default)

        url = mf('url')
        if url is None or url == '':
            raise ValueError('URL is required.')
        width, height, ratio, hash = 0, 0, 0, sha256(url)
        origin_storage_id = get_address_by_key(
            get_s3_key_i2d(meta) if i2d else get_s3_key({'hash': hash})
        )
        try:
            width, height = int(meta.get('width')), int(meta.get('height'))
            ratio = width / height if height != 0 else 0
        except Exception as e:
            pass
        return {
            'url': url,
            'caption': mf('caption', ''),
            'caption_long': mf('caption_long', ''),
            'hash': hash,
            'vector': None,
            'origin_source': mf('source', ''),
            'origin_hash': mf('origin_hash', ''),
            'origin_width': width,
            'origin_height': height,
            'origin_storage_id': origin_storage_id,
            'processed_storage_id': '',
            'processed_width': 0,
            'processed_height': 0,
            'aspect_ratio': ratio,
            'exif': {} if mf('exif') is None else json.loads(mf('exif')),
            'meta': mf('meta', {}),
            'source': name,
            'license': mf('license', ''),
        }

    dataset.name = name
    dataset.map_meta = RULES[name]['fields'] \
        if RULES[name].get('fields').__class__.__name__ == 'function' \
        else map_meta
    dataset.get_s3_key = get_s3_key
    dataset.get_s3_key_i2d = get_s3_key_i2d
    return dataset


def get_datasets():
    SQL = """SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_type = 'BASE TABLE'
        ORDER BY table_name"""
    resp = query(SQL)
    return sorted([table['table_name'][3:] for table in resp])


def count(sql):
    return query(sql)[0]['count']


def get_ratio(a, b, format=True):
    result = a / b if b != 0 else 0
    return f'{round(result * 100, 2)}%' if format else result


def analyze(ds):
    SQL_COUNT = f'SELECT count(id) FROM {ds.get_table_name()}'
    SQL_PROCESSED_FULL = f'{SQL_COUNT} WHERE {SQL_PROCESSED}'
    result = {}
    result['fetched'] = count(SQL_COUNT)
    result['valid'] = count(SQL_PROCESSED_FULL)
    result['valid_ratio'] = get_ratio(result['valid'], result['fetched'])
    result['reprocessed'] = count(
        SQL_PROCESSED_FULL
        + " AND processed_storage_id SIMILAR TO '%processed\\.jpg'"
    )
    result['reprocessed_ratio'] = get_ratio(
        result['reprocessed'], result['valid']
    )
    result['unique'] = count(f'{SQL_PROCESSED_FULL} AND similar_to = 0')
    result['unique_ratio'] = get_ratio(result['unique'], result['valid'])
    result['duplicated'] = count(f'{SQL_PROCESSED_FULL} AND similar_to != 0')
    result['duplicated_ratio'] = get_ratio(
        result['duplicated'], result['valid']
    )
    result['overall_availability'] = get_ratio(
        result['unique'], result['fetched']
    )
    return result


__all__ = [
    'SQL_PROCESSED',
    'analyze',
    'get_datasets',
    'init',
]
