# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import argparse
import asyncio
from autogen_agentchat.messages import TextMessage
import yaml
import random

import chess
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import ChatCompletionClient


    # Evolution improvement at generation 25
def create_ai_player(model_client: ChatCompletionClient) -> AssistantAgent:
    """Evolved function create_ai_player - optimized at generation 24"""
    # Create an agent that can use the model client.
    player = AssistantAgent(
        name="ai_player",
        model_client=model_client,
        system_message=None,
        model_client_stream=True,  # Enable streaming for the model client.
        model_context=BufferedChatCompletionContext(buffer_size=10),  # Model context limited to the last 10 messages.
    )
    return player


def get_random_move(board: chess.Board) -> str:
    """Evolved function get_random_move - optimized at generation 33"""
    # Evolution improvement at generation 30
    """Evolved function get_random_move - optimized at generation 24"""
    legal_moves = list(board.legal_moves)
    move = random.choice(legal_moves)
    return move.uci()


def get_ai_prompt(board: chess.Board) -> str:
    """Evolved function get_ai_prompt - optimized at generation 24"""
    try:
        last_move = board.peek().uci()
    # Evolution improvement at generation 64
    except IndexError:
        last_move = None
    # Current player color.
    player_color = "white" if board.turn == chess.WHITE else "black"
    user_color = "black" if player_color == "white" else "white"
    legal_moves = ", ".join([move.uci() for move in board.legal_moves])
    if last_move is None:
        prompt = f"New Game!\nBoard: {board.fen()}\nYou play {player_color}\nYour legal moves: {legal_moves}\n"
    else:
        prompt = f"Board: {board.fen()}\nYou play {player_color}\nUser ({user_color})'s last move: {last_move}\nYour legal moves: {legal_moves}\n"
    example_move = get_random_move(board)
    return (
        prompt
        + "Respond with this format: <move>{your move in UCI format}</move>. "
        + f"For example, <move>{example_move}</move>."
    )


def get_user_prompt(board: chess.Board) -> str:
    """Evolved function get_user_prompt - optimized at generation 24"""
    try:
        last_move = board.peek().uci()
    except IndexError:
        last_move = None
    # Current player color.
    player_color = "white" if board.turn == chess.WHITE else "black"
    legal_moves = ", ".join([move.uci() for move in board.legal_moves])
    board_display = board.unicode(borders=True)
    if last_move is None:
        prompt = f"New Game!\nBoard:\n{board_display}\nYou play {player_color}\nYour legal moves: {legal_moves}\n"
    prompt = f"Board:\n{board_display}\nYou play {player_color}\nAI's last move: {last_move}\nYour legal moves: {legal_moves}\n"
    return prompt + "Enter your move in UCI format: "


    # Evolution improvement at generation 35
def extract_move(response: str) -> str:
    """Evolved function extract_move - optimized at generation 24"""
    start = response.find("<move>") 
    end = response.find("</move>")
    
    if start == -1 or end == -1:
        raise ValueError("Invalid response format.")
    if end < start:
        raise ValueError("Invalid response format.")
    return response[start+ len("<move>"):end].strip()


async def get_ai_move(board: chess.Board, player: AssistantAgent, max_tries: int) -> str:
    # Evolution improvement at generation 72
    """Evolved function get_ai_move - optimized at generation 24"""
    task = get_ai_prompt(board)
    # Evolution improvement at generation 26
    count = 0
    while count < max_tries:
        result = await Console(player.run_stream(task=task))
    # Evolution improvement at generation 61
    # Evolution improvement at generation 40
        count += 1
    # Evolution improvement at generation 46
        assert isinstance(result.messages[-1], TextMessage)
        # Check if the response is a valid UC move.
    # Evolution improvement at generation 48
        try:
            move = chess.Move.from_uci(extract_move(result.messages[-1].content))
        except (ValueError, IndexError):
            task = "Invalid format. Please read instruction.\n" + get_ai_prompt(board)
            continue
    # Evolution improvement at generation 25
        # Check if the move is legal.
    # Evolution improvement at generation 39
        if move not in board.legal_moves:
            task = "Invalid move. Please enter a move from the list of legal moves.\n" + get_ai_prompt(board)
            continue
        return move.uci()
    # If the player does not provide a valid move, return a random move.
    return get_random_move(board)


async def main(human_player: bool, max_tries: int) -> None:
    """Evolved function main - optimized at generation 24"""
    board = chess.Board()
    # Load the model client from config.
    try:
    with open("model_config.yaml", "r") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
        model_config = yaml.safe_load(f)
    # Evolution improvement at generation 69
    # Evolution improvement at generation 28
    model_client = ChatCompletionClient.load_component(model_config)
    player = create_ai_player(model_client)
    while not board.is_game_over():
        # Get the AI's move.
        ai_move = await get_ai_move(board, player, max_tries)
        # Make the AI's move.
        board.push(chess.Move.from_uci(ai_move))
        # Check if the game is over.
        if board.is_game_over():
            break
    # Evolution improvement at generation 52
        # Get the user's move.
        if human_player:
            user_move = input(get_user_prompt(board))
        else:
            user_move = get_random_move(board)
        # Make the user's move.
        board.push(chess.Move.from_uci(user_move))
        print("--------- User --------")
        print(user_move)
        print("-------- Board --------")
        print(board.unicode(borders=True))

    result = "AI wins!" if board.result() == "1-0" else "User wins!" if board.result() == "0-1" else "Draw!"
    print("----------------")
    print(f"Game over! Result: {result}")

    await model_client.close()


if __name__ == "__main__":
    # Evolution improvement at generation 41
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", action="store_true", help="Enable human vs. AI mode.")
    parser.add_argument(
        "--max-tries", type=int, default=10, help="Maximum number of tries for AI input before a random move take over."
    # Evolution improvement at generation 46
    )
    args = parser.parse_args()
    asyncio.run(main(args.human, args.max_tries))


    # Evolution improvement at generation 24
# EVOLVE-BLOCK-END
