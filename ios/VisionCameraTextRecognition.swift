import Foundation
import VisionCamera
import MLKitVision
import MLKitTextRecognition
import MLKitTextRecognitionChinese
import MLKitTextRecognitionDevanagari
import MLKitTextRecognitionJapanese
import MLKitTextRecognitionKorean
import MLKitCommon
import CoreImage
import UIKit
import AVFoundation
import Accelerate

struct GrayscaleImageData {
    let pixelBuffer: CVPixelBuffer
    let width: Int
    let height: Int
}

@objc(VisionCameraTextRecognition)
public class VisionCameraTextRecognition: FrameProcessorPlugin {

    private var textRecognizer = TextRecognizer()
    private static let latinOptions = TextRecognizerOptions()
    private static let chineseOptions = ChineseTextRecognizerOptions()
    private static let devanagariOptions = DevanagariTextRecognizerOptions()
    private static let japaneseOptions = JapaneseTextRecognizerOptions()
    private static let koreanOptions = KoreanTextRecognizerOptions()
    private var data: [String: Any] = [:]
    
    // Cache CIContext for better performance
    private lazy var ciContext: CIContext = {
        return CIContext(options: [
            .useSoftwareRenderer: false,
            .highQualityDownsample: false
        ])
    }()


    public override init(proxy: VisionCameraProxyHolder, options: [AnyHashable: Any]! = [:]) {
        super.init(proxy: proxy, options: options)
        let language = options["language"] as? String ?? "latin"
        switch language {
        case "chinese":
            self.textRecognizer = TextRecognizer.textRecognizer(options: VisionCameraTextRecognition.chineseOptions)
        case "devanagari":
            self.textRecognizer = TextRecognizer.textRecognizer(options: VisionCameraTextRecognition.devanagariOptions)
        case "japanese":
            self.textRecognizer = TextRecognizer.textRecognizer(options: VisionCameraTextRecognition.japaneseOptions)
        case "korean":
            self.textRecognizer = TextRecognizer.textRecognizer(options: VisionCameraTextRecognition.koreanOptions)
        default:
            self.textRecognizer = TextRecognizer.textRecognizer(options: VisionCameraTextRecognition.latinOptions)
        }
    }


    public override func callback(_ frame: Frame, withArguments arguments: [AnyHashable: Any]?) -> Any {

        let orientation: UIImage.Orientation
        // Check if orientation override is provided in arguments
        if let orientationOverride = arguments?["orientation"] as? Int,
           let visionOrientation = imageOrientationFromInt(orientationOverride) {
            orientation = visionOrientation
        } else {
            // Use default orientation if no override provided
           orientation = getOrientation(orientation: frame.orientation)
        }

        let roi = extractRegionOfInterest(arguments: arguments)
        let visionImage: VisionImage
        
        if let roi = roi {
            // Crop and convert to grayscale if ROI is provided
            guard let croppedPixelBuffer = cropAndConvertToGrayscale(sampleBuffer: frame.buffer, cropRect: roi) else {
                print("Failed to crop and convert image")
                return [:]
            }
            
            // Create VisionImage from CVPixelBuffer via optimized UIImage conversion
            guard let visionImg = createVisionImageFromPixelBuffer(croppedPixelBuffer, orientation: orientation) else {
                print("Failed to create VisionImage from cropped pixel buffer")
                return [:]
            }
            
            visionImage = visionImg
            // Orientation is already set in createVisionImageFromPixelBuffer
        } else {
            // Use the original frame if no ROI
            visionImage = VisionImage(buffer: frame.buffer)
            visionImage.orientation = orientation
        }
        
        do {
            let result = try self.textRecognizer.results(in: visionImage)
            let blocks = VisionCameraTextRecognition.processBlocks(blocks: result.blocks)
            data["resultText"] = result.text
            data["blocks"] = blocks
            if result.text.isEmpty {
                return [:]
            }else{
                return data
            }
        } catch {
            print("Failed to recognize text: \(error.localizedDescription).")
            return [:]
        }
    }

    private func extractRegionOfInterest(arguments: [AnyHashable: Any]?) -> CGRect? {
        guard let arguments = arguments,
              let roiDict = arguments["roi"] as? [String: Any] else {
            return nil
        }
        
        // Handle different number types (Int, Double, CGFloat)
        guard let x = (roiDict["x"] as? NSNumber)?.doubleValue,
              let y = (roiDict["y"] as? NSNumber)?.doubleValue,
              let width = (roiDict["width"] as? NSNumber)?.doubleValue,
              let height = (roiDict["height"] as? NSNumber)?.doubleValue,
              width > 0, height > 0 else {
            return nil
        }
        
        return CGRect(x: CGFloat(x), y: CGFloat(y), width: CGFloat(width), height: CGFloat(height))
    }

    private func cropAndConvertToGrayscale(sampleBuffer: CMSampleBuffer, cropRect: CGRect) -> CVPixelBuffer? {
        // Extract the pixel buffer from the sample buffer
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return nil
        }
        
        // Lock the buffer for reading
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        }
        
        // Get input dimensions
        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
        
        // Ensure crop rectangle is within image bounds and has valid dimensions
        let validCropRect = CGRect(
            x: max(0, min(cropRect.minX, CGFloat(imageWidth - 1))),
            y: max(0, min(cropRect.minY, CGFloat(imageHeight - 1))),
            width: min(max(1, cropRect.width), CGFloat(imageWidth) - max(0, cropRect.minX)),
            height: min(max(1, cropRect.height), CGFloat(imageHeight) - max(0, cropRect.minY))
        )
        
        let outputWidth = Int(validCropRect.width)
        let outputHeight = Int(validCropRect.height)
        
        // Ensure we have valid dimensions
        guard outputWidth > 0 && outputHeight > 0 else {
            return nil
        }
        
        // Create a new pixel buffer for the grayscale output using YUV420 format (optimized for MLKit)
        var newPixelBuffer: CVPixelBuffer?
        let pixelBufferAttributes = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ] as CFDictionary
        
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            outputWidth,
            outputHeight,
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, // YUV420 format like the input
            pixelBufferAttributes,
            &newPixelBuffer
        )
        
        guard status == kCVReturnSuccess, let outputBuffer = newPixelBuffer else {
            return nil
        }
        
        // Lock the output buffer for writing
        CVPixelBufferLockBaseAddress(outputBuffer, [])
        defer {
            CVPixelBufferUnlockBaseAddress(outputBuffer, [])
        }
        
        // Get pixel format to handle different input formats efficiently
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        
        // Handle YUV formats efficiently by directly copying Y plane
        if pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ||
           pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange {
            
            // NV12 format - Y plane is separate, much more efficient
            guard CVPixelBufferGetPlaneCount(pixelBuffer) > 0,
                  CVPixelBufferGetPlaneCount(outputBuffer) > 0,
                  let yPlaneInput = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0),
                  let yPlaneOutput = CVPixelBufferGetBaseAddressOfPlane(outputBuffer, 0),
                  let uvPlaneOutput = CVPixelBufferGetBaseAddressOfPlane(outputBuffer, 1) else {
                return nil
            }
            
            let inputBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
            let outputBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(outputBuffer, 0)
            let uvOutputBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(outputBuffer, 1)
            
            let inputPtr = yPlaneInput.assumingMemoryBound(to: UInt8.self)
            let outputPtr = yPlaneOutput.assumingMemoryBound(to: UInt8.self)
            let uvOutputPtr = uvPlaneOutput.assumingMemoryBound(to: UInt8.self)
            
            // Copy Y plane data efficiently using row-by-row memcpy (much faster than pixel-by-pixel)
            // For even better performance on supported devices, we could use vImage here
            for y in 0..<outputHeight {
                let srcY = Int(validCropRect.minY) + y
                let srcRowOffset = srcY * inputBytesPerRow + Int(validCropRect.minX)
                let dstRowOffset = y * outputBytesPerRow
                
                // Copy entire row at once - MUCH faster than pixel-by-pixel
                let srcPtr = inputPtr.advanced(by: srcRowOffset)
                let dstPtr = outputPtr.advanced(by: dstRowOffset)
                memcpy(dstPtr, srcPtr, outputWidth)
            }
            
            // Fill UV plane with neutral values (128) for better grayscale output
            let uvSize = CVPixelBufferGetHeightOfPlane(outputBuffer, 1) * uvOutputBytesPerRow
            memset(uvOutputPtr, 128, uvSize)
            
        } else {
            // Fallback for other formats - convert to grayscale
            guard let inputBaseAddress = CVPixelBufferGetBaseAddress(pixelBuffer),
                  let yPlaneOutput = CVPixelBufferGetBaseAddressOfPlane(outputBuffer, 0),
                  let uvPlaneOutput = CVPixelBufferGetBaseAddressOfPlane(outputBuffer, 1) else {
                return nil
            }
            
            let inputBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            let outputBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(outputBuffer, 0)
            let uvOutputBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(outputBuffer, 1)
            
            let inputPtr = inputBaseAddress.assumingMemoryBound(to: UInt8.self)
            let outputPtr = yPlaneOutput.assumingMemoryBound(to: UInt8.self)
            let uvOutputPtr = uvPlaneOutput.assumingMemoryBound(to: UInt8.self)
            
            // Convert BGRA to grayscale efficiently
            for y in 0..<outputHeight {
                let srcY = Int(validCropRect.minY) + y
                let srcRowStart = srcY * inputBytesPerRow + Int(validCropRect.minX) * 4
                let dstRowOffset = y * outputBytesPerRow
                
                for x in 0..<outputWidth {
                    let pixelOffset = srcRowStart + x * 4
                    let b = inputPtr[pixelOffset]
                    let g = inputPtr[pixelOffset + 1]
                    let r = inputPtr[pixelOffset + 2]
                    
                    // Optimized grayscale conversion using integer arithmetic
                    let gray = (299 * Int(r) + 587 * Int(g) + 114 * Int(b)) / 1000
                    outputPtr[dstRowOffset + x] = UInt8(min(max(gray, 0), 255))
                }
            }
            
            // Fill UV plane with neutral values
            let uvSize = CVPixelBufferGetHeightOfPlane(outputBuffer, 1) * uvOutputBytesPerRow
            memset(uvOutputPtr, 128, uvSize)
        }
        
        return outputBuffer
    }

      static func processBlocks(blocks:[TextBlock]) -> Array<Any> {
        var blocksArray : [Any] = []
        for block in blocks {
            var blockData : [String:Any] = [:]
            blockData["blockText"] = block.text
            blockData["blockCornerPoints"] = processCornerPoints(block.cornerPoints)
            blockData["blockFrame"] = processFrame(block.frame)
            blockData["lines"] = processLines(lines: block.lines)
            blocksArray.append(blockData)
        }
        return blocksArray
    }

    private static func processLines(lines:[TextLine]) -> Array<Any> {
        var linesArray : [Any] = []
        for line in lines {
            var lineData : [String:Any] = [:]
            lineData["lineText"] = line.text
            lineData["lineLanguages"] = processRecognizedLanguages(line.recognizedLanguages)
            lineData["lineCornerPoints"] = processCornerPoints(line.cornerPoints)
            lineData["lineFrame"] = processFrame(line.frame)
            lineData["elements"] = processElements(elements: line.elements)
            linesArray.append(lineData)
        }
        return linesArray
    }

    private static func processElements(elements:[TextElement]) -> Array<Any> {
        var elementsArray : [Any] = []

        for element in elements {
            var elementData : [String:Any] = [:]
              elementData["elementText"] = element.text
              elementData["elementCornerPoints"] = processCornerPoints(element.cornerPoints)
              elementData["elementFrame"] = processFrame(element.frame)

            elementsArray.append(elementData)
          }

        return elementsArray
    }

    private static func processRecognizedLanguages(_ languages: [TextRecognizedLanguage]) -> [String] {

            var languageArray: [String] = []

            for language in languages {
                guard let code = language.languageCode else {
                    print("No language code exists")
                    break;
                }
                if code.isEmpty{
                    languageArray.append("und")
                }else {
                    languageArray.append(code)

                }
            }

            return languageArray
        }

    private static func processCornerPoints(_ cornerPoints: [NSValue]) -> [[String: CGFloat]] {
        return cornerPoints.compactMap { $0.cgPointValue }.map { ["x": $0.x, "y": $0.y] }
    }

    private static func processFrame(_ frameRect: CGRect) -> [String: CGFloat] {
        let offsetX = (frameRect.midX - ceil(frameRect.width)) / 2.0
        let offsetY = (frameRect.midY - ceil(frameRect.height)) / 2.0

        let x = frameRect.maxX + offsetX
        let y = frameRect.minY + offsetY

        return [
            "x": frameRect.midX + (frameRect.midX - x),
            "y": frameRect.midY + (y - frameRect.midY),
            "width": frameRect.width,
            "height": frameRect.height,
            "boundingCenterX": frameRect.midX,
            "boundingCenterY": frameRect.midY
    ]
    }

    private func getOrientation(orientation: UIImage.Orientation) -> UIImage.Orientation {
        switch orientation {
        case .up:
          return .up
        case .left:
          return .right
        case .down:
          return .down
        case .right:
          return .left
        default:
          return .up
        }
    }
    
    // Helper function to convert integer to image orientation
    private func imageOrientationFromInt(_ orientation: Int) -> UIImage.Orientation? {
        switch orientation {
        case 1: return .up
        case 2: return .down
        case 3: return .left
        case 4: return .right
        case 5: return .upMirrored
        case 6: return .downMirrored
        case 7: return .leftMirrored
        case 8: return .rightMirrored
        default: return nil
        }
    }
    
    // Optimized: Create VisionImage from CVPixelBuffer via UIImage
    private func createVisionImageFromPixelBuffer(_ pixelBuffer: CVPixelBuffer, orientation: UIImage.Orientation) -> VisionImage? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            return nil
        }
        
        let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: orientation)
        return VisionImage(image: uiImage)
    }
}
