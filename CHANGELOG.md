# Changelog

All notable changes to the Number Plate Detection API will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - In Development

### Added
- Version tracking system with dedicated version.py file
- This CHANGELOG.md file to document changes between versions

### Changed
- Updated text rendering with FONT_HERSHEY_COMPLEX for more professional appearance
- Reduced font thickness from 2 to 1 for cleaner, thinner text
- Changed outline color from black to dark gray (70, 70, 70) for a more subtle effect
- Reduced outline thickness and simplified to 4 directions instead of 8
- Adjusted font scale for better proportions

### Fixed
- Improved text readability with better contrast and thinner lines
- Better containment of text within license plate rectangles

## [1.0.0] - 2023-06-15

### Added
- Initial release with license plate detection functionality
- YOLOv8 model integration for accurate license plate detection
- REST API endpoints for image processing and retrieval
- Blue filled rectangle background with white text for license plates
- Text positioning with automatic centering and scaling
- Support for custom text on license plates
- Multiple return formats: JSON, image, and base64
- Image enhancement preprocessing options
- Basic security features including file size validation