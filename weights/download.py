import requests


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id' : id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id':id, 'confirm':token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)
    print('complete')


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)


if __name__ == '__main__':
    file_id = '1AB8ipF9e_t0Du7W79sZtQIOFK9Q9waiQ'
    destination = './vgg16_e086_90.62.pth'
    print('try to download...', end='')
    download_file_from_google_drive(file_id, destination)
