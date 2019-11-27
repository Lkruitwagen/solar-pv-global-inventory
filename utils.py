import requests, json, os, logging

DRIVE_IDS = json.load(open('./drive_ids.json','r'))

def download_file_from_google_drive(_id, destination):

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    logging.info(f'Getting file id: {_id} from google drive')
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : _id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : _id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    logging.info(f'Saving file to {destination}')
    save_response_content(response, destination)

def exists_or_download(fpath):
    if os.path.exists(fpath):
        return fpath
    else:
        logging.info(f'No file found... downloading from drive.')
        download_file_from_google_drive(DRIVE_IDS[fpath], fpath)
        return fpath

def exists_or_mkdir(fpath):
    if not os.path.exists(fpath):
        logging.info(f'Making new path: {fpath}')
        os.makedirs(fpath)