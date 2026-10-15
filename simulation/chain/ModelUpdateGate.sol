// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title ModelUpdateGate — blockchain-verified continual learning for IIoT IDS (DACL)
/// @notice Gateways propose drift-triggered model updates (hashes only; weights stay off-chain).
///         Independent validators evaluate the candidate AND the currently active (parent) model on
///         the same fresh, clean validation data. Per class they measure *detectability*
///         (attack class: share not classified as Normal; Normal: share classified as Normal) and
///         report (i) the known class whose detectability dropped most and (ii) both models' macro
///         detectability over all classes. Policy-as-contract, "net security benefit":
///           accept iff  macro drop        <= maxDetectDropBps   (security does not get worse overall)
///                  and  worst-class drop  <= maxRecallDropBps   (hard floor: no known attack collapses)
///                  and  per-class CUSUM of accepted drops <= maxCumDropBps  (no salami unlearning)
///           escalate (accept, but emit UpdateEscalated for operator review) iff the accepted
///           worst-class drop exceeds softDropBps (a benign but costly trade-off).
///         `threshold` passing attestations activate the model; `threshold` failing ones reject it
///         (the gateway rolls back). Metrics in basis points (1e4 = 100%).
contract ModelUpdateGate {
    enum Status { None, Pending, Accepted, Rejected }

    struct Update {
        address gateway;
        bytes32 modelHash;
        bytes32 parentHash;
        uint64 driftEventId;
        uint32 severityBps;
        uint64 proposedAt;
        uint8 passes;
        uint8 fails;
        Status status;
    }

    address public owner;
    uint8 public threshold;
    uint32 public maxRecallDropBps;
    uint32 public maxDetectDropBps;
    uint32 public maxCumDropBps;
    uint32 public softDropBps;

    mapping(address => bool) public isValidator;
    mapping(address => bool) public isGateway;
    mapping(address => bytes32) public activeModel;        // gateway => accepted model hash
    mapping(address => mapping(uint16 => uint32)) public cumDropBps;  // gateway => class => CUSUM
    mapping(uint256 => mapping(address => bool)) public attested;
    Update[] public updates;

    event GatewayRegistered(address indexed gateway, bytes32 genesisModel);
    event UpdateProposed(uint256 indexed id, address indexed gateway, bytes32 modelHash, bytes32 parentHash,
                         uint64 driftEventId, uint32 severityBps);
    event UpdateAttested(uint256 indexed id, address indexed validator, bool pass, uint16 classId,
                         uint32 candClassBps, uint32 parentClassBps,
                         uint32 candDetectBps, uint32 parentDetectBps);
    event UpdateAccepted(uint256 indexed id, address indexed gateway, bytes32 modelHash);
    event UpdateRejected(uint256 indexed id, address indexed gateway, bytes32 modelHash);
    event UpdateEscalated(uint256 indexed id, address indexed gateway, uint16 classId, int256 dropBps);

    modifier onlyOwner() { require(msg.sender == owner, "not owner"); _; }

    constructor(uint8 _threshold, uint32 _maxRecallDropBps, uint32 _maxDetectDropBps, uint32 _maxCumDropBps,
                uint32 _softDropBps) {
        owner = msg.sender;
        threshold = _threshold;
        maxRecallDropBps = _maxRecallDropBps;
        maxDetectDropBps = _maxDetectDropBps;
        maxCumDropBps = _maxCumDropBps;
        softDropBps = _softDropBps;
    }

    function addValidator(address v) external onlyOwner { isValidator[v] = true; }

    function registerGateway(address g, bytes32 genesisModel) external onlyOwner {
        isGateway[g] = true;
        activeModel[g] = genesisModel;
        emit GatewayRegistered(g, genesisModel);
    }

    function proposeUpdate(bytes32 modelHash, bytes32 parentHash, uint64 driftEventId, uint32 severityBps)
        external returns (uint256 id)
    {
        require(isGateway[msg.sender], "not gateway");
        require(parentHash == activeModel[msg.sender], "parent is not the active model");
        id = updates.length;
        updates.push(Update(msg.sender, modelHash, parentHash, driftEventId, severityBps,
                            uint64(block.timestamp), 0, 0, Status.Pending));
        emit UpdateProposed(id, msg.sender, modelHash, parentHash, driftEventId, severityBps);
    }

    function attest(uint256 id, uint16 classId, uint32 candClassBps, uint32 parentClassBps,
                    uint32 candDetectBps, uint32 parentDetectBps) external {
        require(isValidator[msg.sender], "not validator");
        Update storage u = updates[id];
        require(u.status == Status.Pending, "not pending");
        require(!attested[id][msg.sender], "already attested");
        attested[id][msg.sender] = true;

        int256 dClass = int256(uint256(parentClassBps)) - int256(uint256(candClassBps));
        int256 dDetect = int256(uint256(parentDetectBps)) - int256(uint256(candDetectBps));
        int256 cum = int256(uint256(cumDropBps[u.gateway][classId])) + dClass;
        if (cum < 0) cum = 0;
        bool pass = dClass <= int256(uint256(maxRecallDropBps))
                 && dDetect <= int256(uint256(maxDetectDropBps))
                 && cum <= int256(uint256(maxCumDropBps));
        emit UpdateAttested(id, msg.sender, pass, classId, candClassBps, parentClassBps,
                            candDetectBps, parentDetectBps);

        if (pass) {
            u.passes += 1;
            if (u.passes >= threshold) {
                u.status = Status.Accepted;
                activeModel[u.gateway] = u.modelHash;
                cumDropBps[u.gateway][classId] = uint32(uint256(cum));
                emit UpdateAccepted(id, u.gateway, u.modelHash);
                if (dClass > int256(uint256(softDropBps))) {
                    emit UpdateEscalated(id, u.gateway, classId, dClass);
                }
            }
        } else {
            u.fails += 1;
            if (u.fails >= threshold) {
                u.status = Status.Rejected;
                emit UpdateRejected(id, u.gateway, u.modelHash);
            }
        }
    }

    function updateCount() external view returns (uint256) { return updates.length; }

    function status(uint256 id) external view returns (Status) { return updates[id].status; }
}
