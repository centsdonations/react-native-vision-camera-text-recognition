package com.visioncameratextrecognition

import android.graphics.Point
import android.graphics.Rect
import android.media.Image
import com.facebook.react.bridge.WritableNativeArray
import com.facebook.react.bridge.WritableNativeMap
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions
import com.google.mlkit.vision.text.devanagari.DevanagariTextRecognizerOptions
import com.google.mlkit.vision.text.japanese.JapaneseTextRecognizerOptions
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.mrousavy.camera.frameprocessors.Frame
import com.mrousavy.camera.frameprocessors.FrameProcessorPlugin
import com.mrousavy.camera.frameprocessors.VisionCameraProxy
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min
import java.util.HashMap

data class GrayscaleImageData(
    val buffer: ByteBuffer,
    val width: Int,
    val height: Int
)

class VisionCameraTextRecognitionPlugin(proxy: VisionCameraProxy, options: Map<String, Any>?) :
    FrameProcessorPlugin() {

    private var recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    private val latinOptions = TextRecognizerOptions.DEFAULT_OPTIONS
    private val chineseOptions = ChineseTextRecognizerOptions.Builder().build()
    private val devanagariOptions = DevanagariTextRecognizerOptions.Builder().build()
    private val japaneseOptions = JapaneseTextRecognizerOptions.Builder().build()
    private val koreanOptions = KoreanTextRecognizerOptions.Builder().build()

    init {
        val language = options?.get("language").toString()
        recognizer = when (language) {
            "latin" -> TextRecognition.getClient(latinOptions)
            "chinese" -> TextRecognition.getClient(chineseOptions)
            "devanagari" -> TextRecognition.getClient(devanagariOptions)
            "japanese" -> TextRecognition.getClient(japaneseOptions)
            "korean" -> TextRecognition.getClient(koreanOptions)
            else -> TextRecognition.getClient(latinOptions)
        }
    }

    override fun callback(frame: Frame, arguments: Map<String, Any>?): HashMap<String, Any?>? {
        val data = WritableNativeMap()
        val mediaImage: Image = frame.image

        val rotation = extractOrientationParameter(arguments) ?: frame.imageProxy.imageInfo.rotationDegrees
        val roi = extractRegionOfInterestParameter(arguments)

        val image = if (roi != null) {
            val greyscaleImage = cropAndConvertToGrayscale(mediaImage, roi)
            InputImage.fromByteBuffer(
                greyscaleImage.buffer,
                greyscaleImage.width,
                greyscaleImage.height,
                rotation,
                InputImage.IMAGE_FORMAT_NV21
            )
        } else {
            InputImage.fromMediaImage(mediaImage, rotation)
        }

        val task: Task<Text> = recognizer.process(image)
        try {
            val text: Text = Tasks.await(task)
            if (text.text.isEmpty()) {
                return WritableNativeMap().toHashMap()
            }
            data.putString("resultText", text.text)
            data.putArray("blocks", getBlocks(text.textBlocks))
            return data.toHashMap()
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }
    }

    companion object {
        fun getBlocks(blocks: MutableList<Text.TextBlock>): WritableNativeArray {
            val blockArray = WritableNativeArray()
            blocks.forEach { block ->
                val blockMap = WritableNativeMap().apply {
                    putString("blockText", block.text)
                    putArray("blockCornerPoints", block.cornerPoints?.let { getCornerPoints(it) })
                    putMap("blockFrame", getFrame(block.boundingBox))
                    putArray("lines", getLines(block.lines))
                }
                blockArray.pushMap(blockMap)
            }
            return blockArray
        }

        private fun getLines(lines: MutableList<Text.Line>): WritableNativeArray {
            val lineArray = WritableNativeArray()
            lines.forEach { line ->
                val lineMap = WritableNativeMap().apply {
                    putString("lineText", line.text)
                    putArray("lineCornerPoints", line.cornerPoints?.let { getCornerPoints(it) })
                    putMap("lineFrame", getFrame(line.boundingBox))
                    putArray(
                        "lineLanguages",
                        WritableNativeArray().apply { pushString(line.recognizedLanguage) })
                    putArray("elements", getElements(line.elements))
                }
                lineArray.pushMap(lineMap)
            }
            return lineArray
        }

        private fun getElements(elements: MutableList<Text.Element>): WritableNativeArray {
            val elementArray = WritableNativeArray()
            elements.forEach { element ->
                val elementMap = WritableNativeMap().apply {
                    putString("elementText", element.text)
                    putArray(
                        "elementCornerPoints",
                        element.cornerPoints?.let { getCornerPoints(it) })
                    putMap("elementFrame", getFrame(element.boundingBox))
                }
                elementArray.pushMap(elementMap)
            }
            return elementArray
        }

        private fun getCornerPoints(points: Array<Point>): WritableNativeArray {
            val cornerPoints = WritableNativeArray()
            points.forEach { point ->
                cornerPoints.pushMap(WritableNativeMap().apply {
                    putInt("x", point.x)
                    putInt("y", point.y)
                })
            }
            return cornerPoints
        }

        private fun getFrame(boundingBox: Rect?): WritableNativeMap {
            return WritableNativeMap().apply {
                boundingBox?.let {
                    putDouble("x", it.exactCenterX().toDouble())
                    putDouble("y", it.exactCenterY().toDouble())
                    putInt("width", it.width())
                    putInt("height", it.height())
                    putInt("boundingCenterX", it.centerX())
                    putInt("boundingCenterY", it.centerY())
                }
            }
        }

        private fun extractOrientationParameter(arguments: Map<String, Any>?): Int? {
            if (arguments != null && arguments.containsKey("orientation")) {
                val orientationOverride = arguments["orientation"]
                if (orientationOverride is Int) {
                    // Convert the enum-style orientation to rotation degrees
                    return imageRotationFromOrientationEnum(orientationOverride)
                }
            }
            return null
        }

        private fun extractRegionOfInterestParameter(arguments: Map<String, Any>?): Rect? {
            if (arguments == null) return null
            val roiMap = arguments["roi"] as? Map<*, *> ?: return null

            val x = (roiMap["x"] as? Number)?.toInt() ?: return null
            val y = (roiMap["y"] as? Number)?.toInt() ?: return null
            val width = (roiMap["width"] as? Number)?.toInt() ?: return null
            val height = (roiMap["height"] as? Number)?.toInt() ?: return null

            return Rect(x, y, x + width, y + height)
        }

        fun cropAndConvertToGrayscale(
            image: Image,
            cropRect: android.graphics.Rect
        ): GrayscaleImageData {
            // Get the image planes (Y, U, V for YUV_420_888 format)
            val yPlane = image.planes[0]

            // Original dimensions and strides
            val imageWidth = image.width
            val imageHeight = image.height
            val yRowStride = yPlane.rowStride
            val yPixelStride = yPlane.pixelStride

            // Ensure crop rectangle is within image bounds
            val validCropRect = android.graphics.Rect(
                maxOf(0, cropRect.left),
                maxOf(0, cropRect.top),
                minOf(imageWidth, cropRect.right),
                minOf(imageHeight, cropRect.bottom)
            )

            // Output dimensions are the same as cropped dimensions
            val outputWidth = validCropRect.width()
            val outputHeight = validCropRect.height()

            // Safe buffer size calculation
            val ySize = outputWidth.toLong() * outputHeight.toLong()
            val uvSize = ySize / 2
            val totalSize = ySize + uvSize

            // Check for integer overflow
            if (totalSize > Int.MAX_VALUE) {
                throw IllegalArgumentException("Image dimensions too large: ${outputWidth}x${outputHeight}")
            }

            // Create output buffer for NV21 data
            val outputBuffer = ByteBuffer.allocateDirect(totalSize.toInt())

            // Get the Y plane buffer
            val yBuffer = yPlane.buffer
            val yBufferSize = yBuffer.remaining()

            // Extract and copy Y plane (grayscale)
            var outputPos = 0
            for (y in 0 until outputHeight) {
                val sourceY = validCropRect.top + y
                val sourceRowOffset = sourceY * yRowStride

                for (x in 0 until outputWidth) {
                    val sourceX = validCropRect.left + x
                    val sourceIndex = sourceRowOffset + sourceX * yPixelStride

                    // Safety check for source buffer
                    if (sourceIndex < yBufferSize) {
                        // Safety check for output buffer
                        if (outputPos < ySize.toInt()) {
                            yBuffer.position(sourceIndex)
                            val gray = yBuffer.get()
                            outputBuffer.put(outputPos, gray)
                            outputPos++
                        } else {
                            throw IndexOutOfBoundsException("Y plane write exceeds buffer capacity")
                        }
                    } else {
                        // Default value if out of bounds
                        if (outputPos < ySize.toInt()) {
                            outputBuffer.put(outputPos, 0)
                            outputPos++
                        } else {
                            throw IndexOutOfBoundsException("Y plane write exceeds buffer capacity")
                        }
                    }
                }
            }

            // Fill the UV plane with neutral values (128) for grayscale
            val uvPlaneStart = ySize.toInt()
            val uvPlaneSize = uvSize.toInt()

            // Safety check for UV plane size
            if (uvPlaneStart + uvPlaneSize > totalSize.toInt()) {
                throw IndexOutOfBoundsException("UV plane exceeds buffer capacity")
            }

            // Fill UV plane with neutral values
            for (i in 0 until uvPlaneSize) {
                outputBuffer.put(uvPlaneStart + i, 128.toByte())
            }

            // Reset position to beginning for reading
            outputBuffer.rewind()

            return GrayscaleImageData(
                buffer = outputBuffer,
                width = outputWidth,
                height = outputHeight
            )
        }

        // Helper function to convert orientation enum to rotation degrees
        private fun imageRotationFromOrientationEnum(orientation: Int): Int? {
            return when (orientation) {
                1 -> 0      // .up -> 0 degrees
                2 -> 180    // .down -> 180 degrees
                3 -> 270    // .left -> 270 degrees (counterclockwise from up)
                4 -> 90     // .right -> 90 degrees (clockwise from up)
                5 -> 0      // .upMirrored -> 0 degrees (with horizontal flip, but ML Kit doesn't support mirroring directly)
                6 -> 180    // .downMirrored -> 180 degrees (with horizontal flip)
                7 -> 270    // .leftMirrored -> 270 degrees (with horizontal flip)
                8 -> 90     // .rightMirrored -> 90 degrees (with horizontal flip)
                else -> null
            }
        }
    }
}


