# 发布与许可证检查 — 2026-09-21

## 结论

自有主仓库代码、发布的 head 和小样例现已明确采用 Apache-2.0；三个 App 与
UE 自有部分沿用既有 Apache-2.0。第三方组件、骨干权重和场景资产分别保留原有条款。
目前还不能宣布五个仓库及所有资产均已完成授权整理，主要剩 UE 初始源码来源确认。
本次不设置 GS 清单与论文逐项对齐的任务，也没有创建远端、提交、推送或修改仓库可见性。

## 当前状态

| 对象 | 已确认 | 发布前剩余事项 |
| --- | --- | --- |
| 主仓库 | 已添加根 LICENSE、head/样例许可说明及 Python 包许可元数据 | GitHub 归属、正式源码版本与私密安全反馈入口 |
| Android V4 | 有 Apache-2.0 LICENSE、安全提示和第三方说明 | SDK/依赖条款与安装包说明不能被项目许可证替代 |
| Android V5 | 有 Apache-2.0 LICENSE；已另存 DJI 官方 SDK/样例许可全文和 FFmpeg LGPL-2.1 | 正式 APK 对应依赖通知及源码获取信息 |
| iOS | 有 Apache-2.0 LICENSE；依赖通过 CocoaPods/Swift Packages 引入 | SDK/依赖条款与正式签名安装包分开处理 |
| UE | 自有代码声明 Apache-2.0；已补 MPL 全文，新包携带对应 Eigen 源码；发布树检查通过 | NanoGS 版权头来源、具体发行包与源码版本及引擎分发边界 |

五个仓库当前均没有配置 Git 远端，存在尚未提交的修改。主仓库及三端
SECURITY 文档仍没有可用的私密反馈联系方式；UE 已新增 SECURITY.md 和 HIL 安全说明。
这些是发布入口和维护信息，不是算法缺陷。

本次重新检查了主仓库与三个 App 的当前可发布文件及可达历史。有限模式扫描
未发现匹配的私钥、HF/GitHub token、签名文件或超出项目 50 MiB 阈值的文件：
主仓库 117 个文件；V4 277 个文件 / 260 个历史 blob；V5 2156 / 2105；
iOS 152 / 131。这个扫描不等于完整的凭据或法律审计。

## 重要的上游许可边界

### Pi3X：代码和权重不是同一种许可证

- Pi3 官方代码 LICENSE 为 BSD-3-Clause。
- 官方 README 将 Pi3/Pi3X 模型权重明确列为 CC BY-NC 4.0；Pi3X 的 HF
  模型卡也标注 `license: cc-by-nc-4.0`。
- UAVFF3D 的 Apache-2.0 说明限于其文档和项目页面；README 明确指出数据、
  三维资产、预训练/微调 checkpoint 和第三方模型代码可能使用不同许可。
- 因此，不能因为 GeoFF3D 或 UAVFF3D 根目录存在 Apache-2.0，就推断当前使用的
  UAVFF3D 微调 Pi3X 权重也获得了不受限制的商用授权。

公开指南应写清：自有代码许可不替代骨干权重许可。除非取得覆盖实际 checkpoint
的另行授权，不应承诺使用这些非商业权重的完整流程可直接商用。

本仓库的 5 MB head 是单独的预测器，不包含 Pi3X 骨干参数。不能仅凭它使用了
Pi3X 特征，就自动断言它必然继承 CC BY-NC；head 的单独授权还取决于自有代码、
训练资产及适用协议。本轮已在 `weights/README.md` 和资产清单中明确 head 的
Apache-2.0 许可；该声明不替代外部骨干权重或输入数据的许可。

### DJI：样例源码与 SDK 二进制分开

DJI V5 官方 LICENSE.txt 明确区分：样例代码采用 MIT，SDK 本身受 DJI EULA
约束，并单独提示动态链接 FFmpeg 的 LGPL-2.1 条款。App 自有代码采用 Apache-2.0
不冲突于保留这些独立说明，但不构成对 SDK 或所有安装包依赖的重新授权。
正式安装包需要携带适用的第三方通知及源码获取指引。

### UE：项目代码不等于引擎代码

当前 NanoGS/项目 Source 与 Shader 中发现 102 个含 Epic 版权头的文件。
例如 `Plugins/NanoGS/Shaders/Private/GaussianSplatting.ush` 和
`Plugins/NanoGS/Source/NanoGS/Public/Paged/NanoGSTreeSelector.h`。
这些头可能是生成模板残留，也可能反映代码来源；仅凭头部不能确定。
需要核对来源后分别处理，不能批量删除版权头或据此直接宣称所有文件都属自有代码。

Epic EULA 对普通打包产品、Engine Code 和 Engine Tools 采用不同分发规则。
Engine Tools 包括部分 Editor/Developer 代码及特定开发工具，存在分发渠道限制。
项目含自定义 Editor 模块本身不足以认定违反条款；应区分原创插件代码、
实际随包分发的引擎组件与运行时 PLY 转换工具，确认具体包的适用范围。

UE 打包脚本确实会复制 LICENSE、THIRD_PARTY_NOTICES.md 和 LICENSES/，并非
完全没有通知。Eigen 已在头部保留 MPL-2.0 引用；现已附 MPL 全文，更新后的打包脚本
将对应源码放入 `THIRD_PARTY_SOURCES/eigen3.tar.gz` 并携带获取说明。该义务针对受 MPL 覆盖的组件，
不能泛化为整个 UE 项目必须改用 MPL。

现有 Expo East 包记录了基础提交加未提交修改；正式版本应与固定源码提交对应。

## 建议的协议方案

1. 自有主仓库代码：Apache-2.0，与三个 App 和 UE 自有部分一致。
2. 自有 head：在确认授权权利后明确采用 Apache-2.0；单独说明不包含骨干权重。
3. 第三方源码、SDK 和模型：保留各自许可及版权声明，不套用根许可证。
4. GS/仿真包：保留资产与引擎的适用条款，不把 HF 整仓统一标为代码许可证。
5. 由有权代表作者/所属机构的人确认对自有成果的发布授权，再添加根 LICENSE。

## 本次查阅的原始资料

- [Apache-2.0 官方条款](https://www.apache.org/licenses/LICENSE-2.0.txt)，尤其第 3、4、7、8 条。
- [Pi3 官方代码许可证](https://github.com/yyfz/Pi3/blob/main/LICENSE)。
- [Pi3 官方 README 的 License 表](https://github.com/yyfz/Pi3#-license)。
- [Pi3X 官方模型卡](https://huggingface.co/yyfz233/Pi3X/blob/main/README.md)。
- [UAVFF3D 官方 README](https://github.com/yanxian-ll/UAVFF3D#license)。
- [DJI V5 官方 LICENSE.txt](https://github.com/dji-sdk/Mobile-SDK-Android-V5/blob/dev-sdk-main/LICENSE.txt)。
- [DJI EULA](https://developer.dji.com/policies/eula/)。
- [CC BY-NC 4.0 法律文本](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)。
- [Epic Unreal Engine EULA](https://www.unrealengine.com/en-US/eula/unreal)，第 4–7 节；直连遭遇 403，本次通过网页文本代理读取该官方页面。
- [Mozilla MPL-2.0 官方条款](https://www.mozilla.org/en-US/MPL/2.0/)，第 3.1–3.3 节。

这是面向发布准备的技术与许可核对，不是对全部权利链的法律保证；尚不确定的
来源和授权问题应由维护者或所属机构确认。

## 本轮落地与验证

- 主仓库新增 Apache-2.0 LICENSE、Pi3/COLMAP 保留许可证、head/样例的独立说明，
  并将许可文本纳入 Python wheel。骨干 CC BY-NC 条件单独说明。
- 三个 App 均保存 DJI 官方 SDK/样例许可全文及 FFmpeg LGPL-2.1 全文。
  iOS 另保留 Clipper2 与已安装 DJIWidget 的条款，并移除已不在依赖中的 ZIPFoundation 描述。
- UE 新增 MPL 全文、SECURITY、许可与来源记录；打包流程会附带精确 Eigen 源码。
  临时包验证通过，337 个 Eigen 文件与仓库源码逐文件一致；没有重建或替换现有 HF 包。
- UE 版权头的历史归类为：50 个路径已在旧仓库初始导入中出现，42 个在后续开发中新增，
  10 个在旧仓库没有同路径记录。完整逐文件记录保存在 UE 的
  `Docs/NANOGS_SOURCE_PROVENANCE_20260921.json`；未删除任何版权头。
- Python wheel 构建成功，确认含根 LICENSE、第三方通知及三个保留许可证。
  主仓库与三端的当前文件/历史有限发布扫描通过；UE 发布树检查通过。

剩余需要确认的是 NanoGS 初始导入/改名文件的实际来源、GitHub 仓库归属和名称、
以及可用的私密安全反馈入口。源码公开与正式二进制/场景包发布分别推进。
