//
//  ViewController.swift
//  AudioSR-ios
//
//  Created by kenmaz on 2018/05/16.
//  Copyright © 2018年 kenmaz. All rights reserved.
//

import UIKit
import CoreML

class ViewController: UIViewController {

    @IBAction func buttonDidTouch(_ sender: Any) {

        let lenght = 3
        //let wavData: [Float] = Array(repeating: 0, count: lenght)
        let wavData: [Float] = [0,1,0]
        print(wavData.count)
        let ptr = UnsafeMutablePointer(mutating: wavData)
        do {
            let input = try MLMultiArray(dataPointer: ptr, shape: [1, NSNumber(integerLiteral: lenght), 1], dataType: MLMultiArrayDataType.float32, strides: [1,1,1], deallocator: nil)
            print(input)
            
            let wav = AudioSRInput(wav: input)
            
            do {
                let model = AudioSR()
                let opt = MLPredictionOptions()
                opt.usesCPUOnly = true
                let outFeatures = try model.model.prediction(from: wav, options: opt)
                let output = AudioSROutput(output1: outFeatures.featureValue(for: "output1")!.multiArrayValue!)
                print(output.output1)
            } catch {
                print(error)
            }
        } catch {
            print(error)
        }
    }
    

}

