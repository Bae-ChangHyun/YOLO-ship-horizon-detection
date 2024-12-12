<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Maritime Horizon Detection System</h3>

  <p align="center">
    YOLO 객체 탐지와 Hough 변환을 이용한 수평선 검출 시스템
    <br />
    <a href="https://github.com/your_username/maritime-horizon-detector"><strong>문서 보기 »</strong></a>
    <br />
    <br />
    <a href="https://github.com/your_username/maritime-horizon-detector">데모 보기</a>
    ·
    <a href="https://github.com/your_username/maritime-horizon-detector/issues">버그 제보</a>
    ·
    <a href="https://github.com/your_username/maritime-horizon-detector/issues">기능 요청</a>
  </p>
</div>

<!-- 목차 -->
<details>
  <summary>목차</summary>
  <ol>
    <li>
      <a href="#about-the-project">프로젝트 소개</a>
      <ul>
        <li><a href="#built-with">사용 기술</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">시작하기</a>
      <ul>
        <li><a href="#prerequisites">전제 조건</a></li>
        <li><a href="#installation">설치</a></li>
      </ul>
    </li>
    <li><a href="#usage">사용법</a></li>
    <li><a href="#features">주요 기능</a></li>
    <li><a href="#contributing">기여</a></li>
    <li><a href="#license">라이선스</a></li>
    <li><a href="#contact">연락처</a></li>
  </ol>
</details>

## 프로젝트 소개

이 프로젝트는 해양 환경에서 수평선을 자동으로 감지하고 선박 등의 객체를 식별하는 컴퓨터 비전 시스템입니다. YOLO 객체 탐지와 Hough 변환을 결합하여 효과적인 수평선 검출 및 객체 추적을 구현했습니다.

주요 특징:

- 실시간 수평선 검출
- 선박 및 해양 객체 인식
- 수평선 기준 객체 위치 분류
- 사용자 조정 가능한 파라미터

### 사용 기술

- ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
- ![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
- ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
- ![YOLO](https://img.shields.io/badge/YOLO-00FFFF?style=for-the-badge&logo=YOLO&logoColor=black)

## 시작하기

### 전제 조건

- Python 3.8+
- CUDA 지원 GPU (선택사항)
- pip
  ```sh
  pip install --upgrade pip
  ```
