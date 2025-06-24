import { VisionCameraProxy } from 'react-native-vision-camera';
import type {
  Frame,
  TextRecognitionPlugin,
  TextRecognitionOptions,
  Text,
  ScanTextOptions,
} from './types';

const LINKING_ERROR = `Can't load plugin scanText.Try cleaning cache or reinstall plugin.`;

export function createTextRecognitionPlugin(
  options?: TextRecognitionOptions
): TextRecognitionPlugin {
  const plugin = VisionCameraProxy.initFrameProcessorPlugin('scanText', {
    ...options,
  });
  if (!plugin) {
    throw new Error(LINKING_ERROR);
  }
  return {
    scanText: (frame: Frame, scanOptions?: ScanTextOptions): Text[] => {
      'worklet';
      // @ts-ignore
      return plugin.call(frame, scanOptions ?? {}) as Text[];
    },
  };
}
