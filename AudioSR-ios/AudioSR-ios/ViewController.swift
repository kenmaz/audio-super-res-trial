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

        let lenght = 8192
        let wavData: [Float] = Array(repeating: 0, count: lenght)
        print(wavData.count)
        let ptr = UnsafeMutablePointer(mutating: wavData)
        do {
            let len = NSNumber(integerLiteral: lenght)
            let input = try MLMultiArray(dataPointer: ptr, shape: [1, len, 1], dataType: MLMultiArrayDataType.float32, strides: [1,1,1], deallocator: nil)
            print(input)
            
            do {
                let model = AudioSR()
                let output = try model.prediction(wav: input)
                print(output.output1)
            } catch {
                print(error)
            }
        } catch {
            print(error)
        }
    }
    

}

