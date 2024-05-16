let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
let context = canvas.getContext('2d');
let snap = document.getElementById('snap');
let form = document.getElementById('form');

if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({video: true}).then(function (stream) {
        video.srcObject = stream;
        video.play();
    });
}

window.onload = function() {
    let img = new Image();

    img.onload = function() {
        context.drawImage(img, 0, 0, canvas.width, canvas.height);
    };

    img.src = '/static/image/default_taken_image.jpg'; // replace with your image path
};

document.getElementById('file').addEventListener('change', function () {
    let file = document.getElementById('file');

    convertToBase64(file).then((result) => {
        let img = new Image();
        img.onload = function () {
            let canvasAspectRatio = canvas.width / canvas.height;
            let imgAspectRatio = img.width / img.height;
            let posX, posY, drawWidth, drawHeight;

            if (imgAspectRatio < canvasAspectRatio) {
                drawWidth = canvas.width;
                drawHeight = img.height * (drawWidth / img.width);
                posX = 0;
                posY = (canvas.height - drawHeight) / 2;
            }
            else {
                drawHeight = canvas.height;
                drawWidth = img.width * (drawHeight / img.height);
                posY = 0;
                posX = (canvas.width - drawWidth) / 2;
            }

            context.clearRect(0, 0, canvas.width, canvas.height);
            context.drawImage(img, posX, posY, drawWidth, drawHeight);
        };

        img.src = result;
    });
});

snap.addEventListener("click", function () {
    let input = document.getElementById('image');
    let file = document.getElementById('file');

    if (file.value) {
        convertToBase64(file).then((result) => {
            input.value = result;
        }).then(() => {
            form.submit();
        });
    } else {
        context.drawImage(video, 0, 0, 640, 480)
        window.requestAnimationFrame(function () {
            input.value = canvas.toDataURL();
            form.submit();
        });
    }
});

async function convertToBase64(file) {
    let reader = new FileReader();
    reader.readAsDataURL(file.files[0]);
    return await new Promise((resolve, reject) => {
        reader.onload = () => resolve(reader.result)
    });
}