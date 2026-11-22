"""Builds "Find Soundalikes.shortcut": share a Spotify or Apple Music song link
(or run it with a link copied) and it opens soundalike.yassin.app/find?url=<link>.
No app needed. Sign it with: shortcuts sign --mode anyone -i /tmp/unsigned.shortcut -o "Find Soundalikes.shortcut"
"""
import plistlib
import uuid

OBJ = "\ufffc"  # placeholder replaced by a variable attachment


def var_input():
    return {"Type": "ExtensionInput"}


def action_output(uid, name):
    return {"Type": "ActionOutput", "OutputUUID": uid, "OutputName": name}


def token_string(text, attachments):
    return {
        "Value": {"string": text, "attachmentsByRange": attachments},
        "WFSerializationType": "WFTextTokenString",
    }


def token_attachment(value):
    return {"Value": value, "WFSerializationType": "WFTextTokenAttachment"}


encode_id = str(uuid.uuid4()).upper()
text_id = str(uuid.uuid4()).upper()
prefix = "https://soundalike.yassin.app/find?url="

actions = [
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.urlencode",
        "WFWorkflowActionParameters": {
            "UUID": encode_id,
            "WFEncodeMode": "Encode",
            "WFInput": token_string(OBJ, {"{0, 1}": var_input()}),
        },
    },
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.gettext",
        "WFWorkflowActionParameters": {
            "UUID": text_id,
            "WFTextActionText": token_string(prefix + OBJ, {"{%d, 1}" % len(prefix): action_output(encode_id, "URL Encoded Text")}),
        },
    },
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.openurl",
        "WFWorkflowActionParameters": {
            "WFInput": token_attachment(action_output(text_id, "Text")),
        },
    },
]

shortcut = {
    "WFWorkflowClientVersion": "2700",
    "WFWorkflowMinimumClientVersion": 900,
    "WFWorkflowMinimumClientVersionString": "900",
    "WFWorkflowIcon": {"WFWorkflowIconStartColor": 1440408063, "WFWorkflowIconGlyphNumber": 61440},
    "WFWorkflowTypes": ["ActionExtension"],
    "WFWorkflowInputContentItemClasses": ["WFURLContentItem", "WFStringContentItem"],
    "WFWorkflowHasShortcutInputVariables": True,
    "WFWorkflowNoInputBehavior": {"Name": "WFWorkflowNoInputBehaviorGetClipboard", "Parameters": {}},
    "WFWorkflowImportQuestions": [],
    "WFQuickActionSurfaces": [],
    "WFWorkflowActions": actions,
}

with open("/tmp/unsigned.shortcut", "wb") as handle:
    plistlib.dump(shortcut, handle, fmt=plistlib.FMT_BINARY)
print("wrote /tmp/unsigned.shortcut")
