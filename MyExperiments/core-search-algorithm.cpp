template <bool isPV, searchMode mode>
SearchResults PVS(Board& board, int depth, int alpha, int beta, int ply, SearchContext* ctx, bool cutnode) {
    if constexpr (mode != bench && mode != nodesMode) {
        if (ctx->nodes % 1024 == 0) {
            if constexpr (mode == normal) {
                if (ctx->sw.GetElapsedMS() >= ctx->timeToSearch) {
                    ctx->searchStopped = true;
                    return 0;
                }
            }
        }
    } else if constexpr (mode == nodesMode) {
        if (ctx->nodes > ctx->nodesToGo) {
            ctx->searchStopped = true;
            return 0;
        }
    } else if constexpr (mode == datagen) {
        if (ctx->nodes > DATAGEN::HARD_NODES) {
            ctx->searchStopped = true;
            return 0;
        }
    }

    if (ply > ctx->seldepth)
        ctx->seldepth = ply;


    ctx->pvLine.SetLength(ply);
    if (ply && (IsDraw(board, ctx))) return 0;


    TTEntry* entry = nullptr;
    if (!ctx->excluded)
        entry = ctx->TT.GetRawEntry(board.hashKey);

    const bool ttHit = entry != nullptr
        && entry->hashKey == board.hashKey;

    if constexpr (!isPV) {
        if (ttHit) {
            if (entry->depth >= depth &&
                ((entry->nodeType == PV) ||
                (entry->nodeType == AllNode && entry->score <= alpha) ||
                (entry->nodeType == CutNode && entry->score >= beta))) {

                return SearchResults(entry->score, entry->bestMove);
            }
        }
    }

    if (depth <= 0) return Quiescence<mode>(board, alpha, beta, ply, ctx);

    const int staticEval = NNUE::net.Evaluate(board);
    ctx->ss[ply].eval = staticEval;

    const bool improving = [&]
    {
        if (board.InCheck())
            return false;
        if (ply > 1 && ctx->ss[ply - 2].eval != ScoreNone)
            return staticEval > ctx->ss[ply - 2].eval;
        if (ply > 3 && ctx->ss[ply - 4].eval != ScoreNone)
            return staticEval > ctx->ss[ply - 4].eval;

        return true;
    }();

    if (!board.InCheck() && !ctx->excluded) {
        if (ply) {
            // Reverse Futility Pruning
            int margin = 100 * (depth - improving);
            if (!ttHit && staticEval - margin >= beta && depth < 7) {
                return (beta + (staticEval - beta) / 3);
            }

            // Null Move Pruning
            if (!ctx->doingNullMove && staticEval >= beta) {
                if (depth > 1 && !board.InPossibleZug()) {
                    Board copy = board;
                    copy.MakeMove(Move());

                    const int reduction = 3 + improving;

                    ctx->doingNullMove = true;
                    int score = -PVS<false, mode>(copy, depth - reduction, -beta, -beta + 1, ply + 1, ctx, !cutnode).score;
                    ctx->doingNullMove = false;

                    if (ctx->searchStopped) return 0;
                    if (score >= beta) return score;
                }
            }
        }
    }

    MOVEGEN::GenerateMoves<All>(board);

    SortMoves(board, ply, ctx);

    int score = -inf;
    int nodeType = AllNode;
    SearchResults results(-inf);

    int moveSeen = 0;
    std::array<Move, MAX_MOVES> seenQuiets;
    int seenQuietsCount = 0;

    // For all moves
    for (int i = 0; i < board.currentMoveIndex; i++) {
        Move currMove = board.moveList[i];

        if (ctx->excluded == currMove)
            continue;

        bool notMated = results.score > (-MATE_SCORE + MAX_PLY);

        // Late move pruning
        // If we are near a leaf node we prune moves
        // that are late in the list
        if (!isPV && !board.InCheck() && currMove.IsQuiet() && notMated) {
            int lmpBase = 7;


            int lmpThreshold = lmpBase + 4 * depth;

            if (moveSeen >= lmpThreshold) {
                continue;
            }
        }

        // Futility pruning
        // If our static eval is far below alpha, there is only a small chance
        // that a quiet move will help us so we skip them
        int historyScore = ctx->history[board.sideToMove][currMove.MoveFrom()][currMove.MoveTo()];
        int fpMargin = 100 * depth + historyScore / 32;

        if (!isPV && ply && currMove.IsQuiet()
                && depth <= 5 && staticEval + fpMargin < alpha && notMated) {
            continue;
        }

        Board copy = board;
        bool isLegal = copy.MakeMove(currMove);

        if (!isLegal) continue;

        ctx->ss[ply].pieceType = board.GetPieceType(currMove.MoveFrom());
        ctx->ss[ply].moveTo = currMove.MoveTo();
        ctx->ss[ply].side = board.sideToMove;

        if (copy.positionIndex >= ctx->positionHistory.size()) {
            ctx->positionHistory.resize(copy.positionIndex + 100);
        }
        ctx->positionHistory[copy.positionIndex] = copy.hashKey;
        ctx->nodes++;

        int extension = 0;

        if (ply
            && depth >= 8
            && ttHit
            && currMove == entry->bestMove
            && ctx->excluded == 0
            && entry->depth >= depth - 3
            && entry->nodeType != AllNode)
        {
            const int sBeta = std::max(-inf + 1, entry->score - depth * 2);
            const int sDepth = (depth - 1) / 2;

            ctx->excluded = currMove;
            const int singularScore = PVS<false, mode>(board, sDepth, sBeta-1, sBeta, ply, ctx, cutnode).score;
            ctx->excluded = Move();

            if (singularScore < sBeta)
                extension = 1;
        }

        // PVS SEE
        int SEEThreshold = currMove.IsQuiet() ? -80 * depth : -30 * depth * depth;

        if (ply && depth <= 10 && !SEE(board, currMove, SEEThreshold))
            continue;

        int reductions = GetReductions<isPV>(board, currMove, depth, moveSeen, ply, cutnode, ctx);

        int newDepth = depth + copy.InCheck() - 1 + extension;

        // First move (suspected PV node)
        if (!moveSeen) {
            // Full search
            if constexpr (isPV) {
                score = -PVS<isPV, mode>(copy, newDepth, -beta, -alpha, ply + 1, ctx, false).score;
            } else {
                score = -PVS<isPV, mode>(copy, newDepth, -beta, -alpha, ply + 1, ctx, !cutnode).score;
            }
        } else if (reductions) {
            // Null-window search with reductions
            score = -PVS<false, mode>(copy, newDepth - reductions, -alpha-1, -alpha, ply + 1, ctx, true).score;

            if (score > alpha) {
                // Null-window search now without the reduction
                score = -PVS<false, mode>(copy, newDepth, -alpha-1, -alpha, ply + 1, ctx, !cutnode).score;
            }
        } else {
            // Null-window search
            score = -PVS<false, mode>(copy, newDepth, -alpha-1, -alpha, ply + 1, ctx, !cutnode).score;
        }

        // Check if we need to do full window re-search
        if (moveSeen && score > alpha && score < beta) {
            score = -PVS<isPV, mode>(copy, newDepth, -beta, -alpha, ply + 1, ctx, false).score;
        }

        moveSeen++;

        if (ctx->searchStopped) return 0;

        if (currMove != 0 && currMove.IsQuiet()) {
            seenQuiets[seenQuietsCount] = currMove;
            seenQuietsCount++;
        }

        // Fail high (beta cutoff)
        if (score >= beta) {
            if (!currMove.IsCapture()) {
                ctx->killerMoves[1][ply] = ctx->killerMoves[0][ply];
                ctx->killerMoves[0][ply] = currMove;

                int bonus = 300 * depth - 250;

                ctx->history.Update(board.sideToMove, currMove, bonus);

                if (ply > 0) {
                    int prevType = ctx->ss[ply-1].pieceType;
                    int prevTo = ctx->ss[ply-1].moveTo;
                    int pieceType = ctx->ss[ply].pieceType;
                    int to = ctx->ss[ply].moveTo;
                    bool otherColor = ctx->ss[ply-1].side;

                    ctx->conthist.Update(board.sideToMove, otherColor, prevType, prevTo, pieceType, to, bonus);

                    // Malus
                    for (int moveIndex = 0; moveIndex < seenQuietsCount - 1; moveIndex++) {
                        int pieceType = board.GetPieceType(seenQuiets[moveIndex].MoveFrom());
                        int to = seenQuiets[moveIndex].MoveTo();

                        ctx->conthist.Update(board.sideToMove, otherColor, prevType, prevTo, pieceType, to, -bonus);
                    }

                    if (ply > 1) {
                        prevType = ctx->ss[ply-2].pieceType;
                        prevTo = ctx->ss[ply-2].moveTo;
                        otherColor = ctx->ss[ply-2].side;

                        ctx->conthist.Update(board.sideToMove, otherColor, prevType, prevTo, pieceType, to, bonus);

                        // Malus
                        for (int moveIndex = 0; moveIndex < seenQuietsCount - 1; moveIndex++) {
                            int pieceType = board.GetPieceType(seenQuiets[moveIndex].MoveFrom());
                            int to = seenQuiets[moveIndex].MoveTo();

                            ctx->conthist.Update(board.sideToMove, otherColor, prevType, prevTo, pieceType, to, -bonus);
                        }

                    }
                }


                // Malus
                for (int moveIndex = 0; moveIndex < seenQuietsCount - 1; moveIndex++) {
                    ctx->history.Update(board.sideToMove, seenQuiets[moveIndex], -bonus);
                }
            }

            if (!ctx->excluded)
                ctx->TT.WriteEntry(board.hashKey, depth, score, CutNode, currMove);
            return score;
        }

        results.score = std::max(score, results.score);

        if (score > alpha) {
            nodeType = PV;
            alpha = score;
            results.bestMove = currMove;
            ctx->pvLine.SetMove(ply, currMove);
        }
    }

    if (moveSeen == 0) {
        if (board.InCheck()) { // checkmate
            return -MATE_SCORE + ply;
        } else { // stalemate
            return 0;
        }
    }

    if (ctx->searchStopped) return 0;
    if (!ctx->excluded)
        ctx->TT.WriteEntry(board.hashKey, depth, results.score, nodeType, results.bestMove);
    return results;
}
