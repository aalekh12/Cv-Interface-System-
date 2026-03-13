import http from 'k6/http';

const imageData = open('./cat.jpg', 'b');

export const options = {
    vus: 10,
    duration: '30s',
};

export default function () {
    const url = 'http://0.0.0.0:9000/predict';

    const payload = {
        image: http.file(imageData, 'cat.jpg', 'image/jpeg'),
    };

    const params = {
        headers: {
            // k6 automatically sets multipart boundary
            // but this helps ensure FastAPI treats it correctly
            'Accept': 'application/json',
        },
    };

    http.post(url, payload, params);
}