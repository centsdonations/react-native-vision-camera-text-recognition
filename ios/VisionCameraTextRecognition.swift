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
            guard let grayscaleData = cropAndConvertToGrayscale(sampleBuffer: frame.buffer, cropRect: roi) else {
                print("Failed to crop and convert image")
                return [:]
            }
            
            // Create VisionImage from the cropped grayscale buffer using a UIImage as intermediary
            let ciImage = CIImage(cvPixelBuffer: grayscaleData.pixelBuffer)
            let context = CIContext(options: nil)
            guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
                return [:]
            }
            let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: orientation)
            visionImage = VisionImage(image: uiImage)
        } else {
            // Use the original frame if no ROI
            visionImage = VisionImage(buffer: frame.buffer)
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
              let roiDict = arguments["roi"] as? [String: Any],
              let x = roiDict["x"] as? CGFloat,
              let y = roiDict["y"] as? CGFloat,
              let width = roiDict["width"] as? CGFloat,
              let height = roiDict["height"] as? CGFloat else {
            return nil
        }
        
        return CGRect(x: x, y: y, width: width, height: height)
    }

    private func cropAndConvertToGrayscale(sampleBuffer: CMSampleBuffer, cropRect: CGRect) -> GrayscaleImageData? {
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
        
        // Ensure crop rectangle is within image bounds
        let validCropRect = CGRect(
            x: max(0, cropRect.minX),
            y: max(0, cropRect.minY),
            width: min(CGFloat(imageWidth) - cropRect.minX, cropRect.width),
            height: min(CGFloat(imageHeight) - cropRect.minY, cropRect.height)
        )
        
        let outputWidth = Int(validCropRect.width)
        let outputHeight = Int(validCropRect.height)
        
        // Create a new pixel buffer for the grayscale output
        var newPixelBuffer: CVPixelBuffer?
        let pixelBufferAttributes = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary
        
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            outputWidth,
            outputHeight,
            kCVPixelFormatType_OneComponent8, // 8-bit grayscale
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
        
        // Get the base address of both buffers
        guard let inputBaseAddress = CVPixelBufferGetBaseAddress(pixelBuffer),
              let outputBaseAddress = CVPixelBufferGetBaseAddress(outputBuffer) else {
            return nil
        }
        
        // Get strides for each buffer
        let inputBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let outputBytesPerRow = CVPixelBufferGetBytesPerRow(outputBuffer)
        
        // Get pixel format
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        
        // Extract the Y plane (grayscale) from YUV format or convert RGB to grayscale
        for y in 0..<outputHeight {
            let srcY = Int(validCropRect.minY) + y
            let srcRowOffset = srcY * inputBytesPerRow
            let dstRowOffset = y * outputBytesPerRow
            
            for x in 0..<outputWidth {
                let srcX = Int(validCropRect.minX) + x
                
                var grayValue: UInt8 = 0
                
                // Different handling based on pixel format
                if pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ||
                   pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange {
                    // This is NV12 format (commonly used in iOS camera frames)
                    // Y plane is the first plane and contains grayscale values directly
                    
                    if CVPixelBufferGetPlaneCount(pixelBuffer) > 0 {
                        // Get the Y plane
                        let yPlaneBaseAddress = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0)
                        let yPlaneBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
                        
                        // Extract the Y value
                        let yOffset = srcY * yPlaneBytesPerRow + srcX
                        grayValue = yPlaneBaseAddress!.advanced(by: yOffset).load(as: UInt8.self)
                    }
                } else if pixelFormat == kCVPixelFormatType_32BGRA {
                    // For BGRA format, need to convert RGB to grayscale
                    let pixelOffset = srcRowOffset + srcX * 4
                    let b = inputBaseAddress.advanced(by: pixelOffset).load(as: UInt8.self)
                    let g = inputBaseAddress.advanced(by: pixelOffset + 1).load(as: UInt8.self)
                    let r = inputBaseAddress.advanced(by: pixelOffset + 2).load(as: UInt8.self)
                    
                    // Standard RGB to grayscale conversion
                    // Y = 0.299*R + 0.587*G + 0.114*B
                    grayValue = UInt8(min(max(0.299 * Double(r) + 0.587 * Double(g) + 0.114 * Double(b), 0), 255))
                }
                
                // Write the grayscale value to the output buffer
                let outputOffset = dstRowOffset + x
                outputBaseAddress.advanced(by: outputOffset).storeBytes(of: grayValue, as: UInt8.self)
            }
        }
        
        return GrayscaleImageData(
            pixelBuffer: outputBuffer,
            width: outputWidth,
            height: outputHeight
        )
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
}
