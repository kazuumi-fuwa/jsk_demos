# G1 Tablecloth Manipulation

Unitree G1によるテーブルクロスの認識・把持・展開を目的としたROS 2プロジェクト。

## 開発環境

- Ubuntu 24.04
- ROS 2 Jazzy
- MoveIt 2
- Grounding DINO + SAM2
- Intel RealSense RGB-D Camera
- Unitree G1（Inspire Hand）

## ディレクトリ構成

| ディレクトリ | 内容 |
|---|---|
| `perception/` | テーブルクロスの認識、四隅検出、3D座標推定 |
| `motion/` | G1の軌道計画、実機実行インターフェース、ハンド制御 |
| `tools/` | rosbag解析、画像抽出、カメラ較正 |
| `tests/` | 個別機能の検証 |
| `legacy/` | 過去の実験コード |

既存のスクリプトは互換性のため、一部をルートにも残している。

## 実装済みの機能

- Grounding DINOによるテーブルクロス候補検出
- SAM2によるマスク生成・追跡
- OpenCVによる四隅の検出
- RGB画像とDepth画像のタイムスタンプ対応付け
- RGB-Dからの3D位置推定コード
- MoveItによる右腕のPre-grasp / Approach軌道計画
- Inspire Hand開閉用のアクション通信コード

## 動作確認済み

- Grounding DINO + SAM2による画像認識
- 四隅の2D検出
- Mock座標を用いたMoveItの軌道計画
- 2段階の連続軌道計画
- ハンド開閉コマンドのDRY RUN

## 未検証

- G1搭載カメラのRGB-Depth較正とTF
- 実カメラからの3D把持候補を用いた軌道計画
- 手首から指先への位置変換
- テーブル・床の障害物設定
- 実機コントローラによる軌道実行
- 実機ハンドの開閉
- 布の把持・持ち上げ・展開

## シミュレーション

ROS 2環境を読み込んでMoveItを起動する。

```bash
source /opt/ros/jazzy/setup.bash
source ~/semi_ws/install/setup.bash

ros2 launch g1_moveit_config demo.launch.py hand_type:=inspire_dfq
```

別ターミナルで：

```bash
cd ~/semi_ws/src/jsk_demos/jsk_2026_10_semi/scripts/fuwa/motion

python3 g1_plan_approach.py --mode mock --plan
python3 g1_execute_pregrasp.py --mode mock
python3 g1_hand_control.py --hand right --action open
```

いずれも実機を動かさないテスト。

## 実機テスト前の注意

現状の腕の姿勢と把持位置は検証用である。カメラ較正、TF、衝突環境、コントローラ、緊急停止、手先姿勢を確認するまで、実機実行オプションは使用しない。

## 次の作業

1. G1搭載カメラのCameraInfo・Depth・TFを確認
2. 実カメラから3D把持位置を計算
3. MoveItの軌道実行先を確認
4. 実機ハンドの開閉を検証
5. 布のPre-grasp → Approach → Grasp → Liftを段階的に実装
