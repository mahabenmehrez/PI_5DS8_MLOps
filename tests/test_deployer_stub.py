from agents.deployer.stub import DeployerStub
from shared.messages import Message, MessageType


def test_rollout_sequence():
    stub = DeployerStub()
    msgs = stub.run_rollout("2")
    stages = [m.payload["to_stage"] for m in msgs]
    assert stages == ["shadow", "10%", "50%"]


def test_scenario_g_rolls_back():
    stub = DeployerStub(mode="scenario_g")
    msgs = stub.run_rollout("2")
    assert msgs[-1].kind == "deployment.rolled_back"
    assert msgs[-1].payload["automatic"] is True


def test_answer_deploy_candidate():
    req = Message(
        type=MessageType.REQUEST,
        kind="request.deploy_candidate",
        sender="trainer",
        to=["deployer"],
        topic="model:wine-quality@2",
        level=1,
        summary="Please deploy candidate",
        payload={"version": "2"},
    )
    ans = DeployerStub().answer_deploy_candidate(req)
    assert ans.payload["decision"] == "accepted"
    assert ans.reply_to == req.id


def test_query_deployment_changed():
    assert DeployerStub().answer_query_deployment_changed("x") == {"changes": []}