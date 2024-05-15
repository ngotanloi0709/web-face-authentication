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

snap.addEventListener("click", function () {
    let input = document.getElementById('image');
    let file = document.getElementById('file');

    if (file.value) {
        convertToBase64().then((result) => {
            input.value = result;
        }).then(() => {
            form.submit();
        });
    } else {
        context.drawImage(video, 0, 0, 640, 480)
        window.requestAnimationFrame(function() {
            input.value = canvas.toDataURL();
            form.submit();
        });
    }
});

async function convertToBase64() {
    var reader = new FileReader();
    reader.readAsDataURL(file.files[0]);
    result = await new Promise((resolve, reject) => {
        reader.onload = () => resolve(reader.result)
    });
    return result;
}