using UnityEngine;
using UnityEngine.UI;

namespace CV_AR.Camera
{
    [RequireComponent(typeof(RawImage))]
    public class BackgroundCamera : MonoBehaviour
    {
        private WebCamTexture webCamTexture;
        private RawImage rawImage;
        private AspectRatioFitter fitter;

        void Start()
        {
            rawImage = GetComponent<RawImage>();
            
            // AspectRatioFitter가 없다면 자동 추가 (비율 유지를 위해)
            fitter = GetComponent<AspectRatioFitter>();
            if (fitter == null)
            {
                fitter = gameObject.AddComponent<AspectRatioFitter>();
            }
            fitter.aspectMode = AspectRatioFitter.AspectMode.EnvelopeParent;

            // 기본 웹캠 장치 가져오기
            WebCamDevice[] devices = WebCamTexture.devices;
            if (devices.Length > 0)
            {
                // 해상도는 가볍게 설정
                webCamTexture = new WebCamTexture(devices[0].name, 1280, 720, 30);
                rawImage.texture = webCamTexture;
                webCamTexture.Play();
            }
            else
            {
                Debug.LogError("[BackgroundCamera] 웹캠을 찾을 수 없습니다.");
            }
        }

        void Update()
        {
            if (webCamTexture != null && webCamTexture.didUpdateThisFrame)
            {
                // 웹캠 비율에 맞게 UI 이미지 비율 조정
                float ratio = (float)webCamTexture.width / (float)webCamTexture.height;
                fitter.aspectRatio = ratio;

                // 좌우 반전 및 회전 보정 (맥북 웹캠 등 기기에 따라 다를 수 있음)
                float scaleY = webCamTexture.videoVerticallyMirrored ? -1f : 1f;
                rawImage.rectTransform.localScale = new Vector3(1f, scaleY, 1f);

                int orient = -webCamTexture.videoRotationAngle;
                rawImage.rectTransform.localEulerAngles = new Vector3(0, 0, orient);
            }
        }

        void OnDestroy()
        {
            if (webCamTexture != null)
            {
                webCamTexture.Stop();
            }
        }
    }
}
